/*
   FALCON - The Falcon Programming Language.
   FILE: messagequeue.cpp

   Message queue
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 20 Feb 2013 11:15:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "engine/messagequeue.cpp"

#include <falcon/messagequeue.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/cm/fence.h>
#include <falcon/stdhandlers.h>
#include <falcon/contextmanager.h>

#include <map>

namespace Falcon
{

class MessageQueue::MQFence: public Ext::SharedFence
{
public:
   MQFence( ContextManager* mgr, const Class* owner, int32 fenceCount ):
      Ext::SharedFence(mgr, owner, fenceCount, true)
   {}

   virtual ~MQFence() {}

   MQFence* m_next;
};

class MessageQueue::Private
{
public:
   typedef std::map<VMContext*, Token*> ContextMap;
   ContextMap m_contexts;

   Private() {}
   ~Private() {}
};


class MessageQueue::Token
{
public:
   Item m_item;
   String m_evtName;

   Token* m_next;
   uint32 m_readCount;

   // used to count how many tokens we're recycling.
   int32 m_sequence;
};

MessageQueue::MessageQueue( ContextManager* mgr, const String& name ):
   Shared( mgr, Engine::handlers()->messageQueueClass() ),
   m_ctxWeakRef( this ),
   m_name(name),
   m_version(0),
   m_bSignal(true),
   m_firstToken(0),
   m_lastToken(0),
   m_recycleBin(0),
   m_recycleCount(0)
{
   _p = new Private;
   // create a dummy token
   m_lastToken = m_firstToken = new Token;

   m_firstFence = 0;
   m_firstToken->m_sequence = 0;
   m_firstToken->m_readCount = 0;
   m_firstToken->m_next = 0;

   // We're a context-specific resource.
   // Wait and consumption operations can behave differently on
   // different contexts.
   m_bContextSpec = true;
}


MessageQueue::~MessageQueue()
{
   delete _p;

   Token * tk = m_firstToken;
   while (tk != 0)
   {
      Token* old = tk;
      tk = tk->m_next;
      delete old;
   }

   tk = m_recycleBin;
   while (tk != 0)
   {
      Token* old = tk;
      tk = tk->m_next;
      delete old;
   }
}


void MessageQueue::gcMark( uint32 n )
{
   if( currentMark() != n )
   {
      Shared::gcMark(n);

      m_mtx.lock();
      uint32 version = m_version;
      Token* token = m_firstToken->m_next;
      while( token != 0 )
      {
         Class* cls = 0;
         void* data = 0;
         bool toMark = token->m_item.asClassInst(cls, data);
         if( toMark )
         {
            m_mtx.unlock();

            cls->gcMarkInstance(data,n);

            m_mtx.lock();
            // restart if version is changed.
            if( version != m_version )
            {
               version = m_version;
               token =  m_firstToken;
            }
         }

         token = token->m_next;
      }

      // then mark all the fences
      MQFence* mqf =  m_firstFence;
      while( mqf != 0 )
      {
         mqf->gcMark(n);
         mqf = mqf->m_next;
      }
      m_mtx.unlock();
   }
}


int32 MessageQueue::subscribers() const
{
   m_mtx.lock();
   int32 subs = (int32) _p->m_contexts.size();
   m_mtx.unlock();

   return subs;
}


Shared* MessageQueue::subscriberWaiter( int count )
{
   MQFence* mqf = new MQFence( notifyTo(), Engine::handlers()->sharedClass(), count );

   m_mtx.lock();
   int32 signals = (int32) _p->m_contexts.size();
   if( signals < count )
   {
      mqf->m_next = m_firstFence;
      m_firstFence = mqf;
   }
   m_mtx.unlock();

   // tell the fence how many subscribers we have now.
   mqf->signal(signals);
   FALCON_GC_HANDLE(mqf);
   return mqf;
}


int32 MessageQueue::consumeSignal( VMContext* ctx, int32 )
{
   // check if we have a message for that target.
   m_mtx.lock();
   Private::ContextMap::iterator pos = _p->m_contexts.find( ctx );
   if( pos == _p->m_contexts.end() || pos->second->m_next == 0 )
   {
      m_mtx.unlock();
      return 0;
   }

   m_mtx.unlock();

   return 1;
}


int32 MessageQueue::lockedConsumeSignal( VMContext* target, int32 count )
{
   return MessageQueue::consumeSignal(target,count);
}


void MessageQueue::onWaiterAdded(VMContext* ctx)
{
   subscribe(ctx);
}


bool MessageQueue::subscribe(VMContext* ctx)
{
   m_mtx.lock();
   Private::ContextMap::iterator pos = _p->m_contexts.find( ctx );
   if( pos != _p->m_contexts.end() )
   {
      // already subscribed
      m_mtx.unlock();

      return false;
   }
   _p->m_contexts[ctx] = m_firstToken;

   // signal all the fences
   MQFence* mqf =  m_firstFence;
   MQFence* prev = 0;
   while( mqf != 0 )
   {
      // ready to disengage -- check it before signaling,
      // or the waiter may get the signals.
      if ( mqf->level() <= 1 )
      {
         // remove the current fence
         if( prev != 0 )
         {
            prev->m_next = mqf->m_next;
         }
         else
         {
            m_firstFence = m_firstFence->m_next;
         }
      }
      else {
         prev = mqf;
      }
      mqf->signal(1);

      mqf = mqf->m_next;
   }

   // Add
   m_mtx.unlock();

   ctx->registerOnTerminate(&this->m_ctxWeakRef);
   ctx->incref();
   return true;
}


bool MessageQueue::unsubscribe(VMContext* ctx)
{
   m_mtx.lock();
   Private::ContextMap::iterator pos = _p->m_contexts.find( ctx );
   if( pos == _p->m_contexts.end() )
   {
      // already subscribed
      m_mtx.unlock();

      return false;
   }
   _p->m_contexts.erase(pos);

   // Add
   m_mtx.unlock();

   // DO NOT DECREF THE UNREAD MESSAGES
   // -- the next time a newer message is add and read, all the messages
   // -- that the subscriber didn't read will be disposed of.
   // -- However, this might be a memory hungry and bloatsome solution;
   // -- in case the queue grows out of control, it might be necessary
   // -- to decref the messages that the subscriber didn't read yet
   // -- when it was removed.
   ctx->unregisterOnTerminate(&this->m_ctxWeakRef);
   ctx->decref();
   return true;
}


bool MessageQueue::send( const Item& message )
{
   return sendEvent("", message);
}


bool MessageQueue::sendEvent( const String& eventName, const Item& message )
{
   m_mtx.lock();
   // we don't support late subscription. Incoming messages are lost.
   if( _p->m_contexts.empty() )
   {
      m_mtx.unlock();
      return false;
   }

   Token* token = allocToken();
   m_version++;

   bool bSignal = m_bSignal;
   m_bSignal = false;
   m_lastToken->m_next = token;
   token->m_sequence = m_lastToken->m_sequence + 1;
   // we're pretty sure that the message comes from the stack.
   token->m_item = message;
   // Set the reader count to the number of current subscribers
   token->m_readCount = _p->m_contexts.size();

   token->m_evtName.size(0);
   token->m_evtName.append(eventName);
   m_lastToken = token;
   m_mtx.unlock();

   // just wake up existing subcribers, if any.
   if ( bSignal ) {
      notifyTo()->onSharedSignaled(this);
   }

   return true;
}


bool MessageQueue::get( VMContext* ctx, Item& msg )
{
   String temp;
   return getEvent( ctx, temp, msg );
}


bool MessageQueue::getEvent( VMContext* ctx, String& eventName, Item& msg )
{
   m_mtx.lock();
   Private::ContextMap::iterator pos = _p->m_contexts.find( ctx );
   if( pos == _p->m_contexts.end() )
   {
      // not subscribed
      m_mtx.unlock();
      return false;
   }

   Token* token = pos->second;
   if (token->m_next == 0)
   {
      // no new messages
      m_mtx.unlock();
      return false;
   }

   // store the message.
   Token* next = token->m_next;
   msg = next->m_item;
   eventName = next->m_evtName;

   // last token for this context?
   if( next->m_next == 0 )
   {
      m_bSignal = true;  // signal on arrival of new messages.
   }

   // update the pointer for this context
   pos->second = next;

   // update the token read count.
   // Actually, the token was was read previously,
   // but it's ok if we keep the convention of marking the previous element.
   next->m_readCount--;

   // is the token abandoned by all subscribers?
   if( token != m_firstToken && next->m_readCount == 0 )
   {
      // recycle all the previous tokens -- the head is our dummy, so we go next
      recycleTokens(m_firstToken->m_next, token);
      // advance one
      m_firstToken->m_next = next;
      // tell the list is changed.
      m_version++;
   }

   m_mtx.unlock();
   return true;
}


bool MessageQueue::peek( VMContext* ctx, Item& msg )
{
   m_mtx.lock();
   Private::ContextMap::iterator pos = _p->m_contexts.find( ctx );
   if( pos == _p->m_contexts.end() )
   {
      // not subscribed
      m_mtx.unlock();
      return false;
   }

   Token* token = pos->second;
   if (token->m_next == 0)
   {
      // no new messages
      m_mtx.unlock();
      return false;
   }

   msg = pos->second->m_item;
   m_mtx.unlock();

   return true;
}


MessageQueue::Token* MessageQueue::allocToken()
{
   m_mtxRecycle.lock();
   Token* tk;
   if( m_recycleBin != 0 )
   {
      tk = m_recycleBin;
      m_recycleBin = m_recycleBin->m_next;
      // the recycle count can fall below zero if we
      // underestimated the count of recycled tokens.
      if ( m_recycleCount > 0 )
      {
         m_recycleCount--;
      }
      m_mtxRecycle.unlock();
   }
   else {
      m_mtxRecycle.unlock();
      tk = new Token;
   }

   tk->m_readCount = 0;
   tk->m_next = 0;
   return tk;
}


void MessageQueue::onWakeupComplete()
{
   lockedConsumeSignal(0, 1);
}


void MessageQueue::recycleToken( Token* token )
{
   m_mtxRecycle.lock();
   if( m_recycleCount < MAX_TOKEN_RECYCLE_COUNT )
   {
      token->m_next = m_recycleBin;
      m_recycleBin = token;
      m_recycleCount++;
      m_mtxRecycle.unlock();
   }
   else {
      m_mtxRecycle.unlock();
      delete token;
   }
}


void MessageQueue::recycleTokens( Token* first, Token* last )
{
   int32 count = last->m_sequence - first->m_sequence+1;
   if( count <= 0 )
   {
      // we might underestimate the token reserve a bit at rollover,
      // but that's not a problem.
      count = 1;
   }

   m_mtxRecycle.lock();
   if( m_recycleCount + count <= MAX_TOKEN_RECYCLE_COUNT )
   {
      last->m_next = m_recycleBin;
      m_recycleBin = first;
      m_recycleCount += count;
      m_mtxRecycle.unlock();
   }
   else {
      m_mtxRecycle.unlock();
      Token* tk = first;
      while (tk != last)
      {
         Token* old = tk;
         tk = tk->m_next;
         delete old;
      }
      delete last;
   }
}

MessageQueue* MessageQueue::clone() const
{
   const_cast<MessageQueue*>(this)->incref();
   return const_cast<MessageQueue*>(this);
}


MessageQueue::CtxWeakRef::CtxWeakRef( MessageQueue* owner ):
         m_owner(owner)
{
}

MessageQueue::CtxWeakRef::~CtxWeakRef() {}

void MessageQueue::CtxWeakRef::onTerminate( VMContext* ctx )
{
   // unregister without unregistering in the context.
   m_owner->m_mtx.lock();
   Private::ContextMap::iterator pos = m_owner->_p->m_contexts.find( ctx );
   if( pos != m_owner->_p->m_contexts.end() )
   {
      m_owner->_p->m_contexts.erase(pos);
   }
   m_owner->m_mtx.unlock();
}


}

/* end of shared.cpp */
