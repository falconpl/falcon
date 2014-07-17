/*
   FALCON - The Falcon Programming Language.
   FILE: eventcurrier.cpp


   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/eventcourier.cpp"
#include <falcon/eventcourier.h>
#include <falcon/classes/classeventcourier.h>
#include <falcon/stdhandlers.h>
#include <falcon/mt.h>
#include <falcon/itemarray.h>
#include <falcon/error.h>
#include <falcon/vmcontext.h>
#include <falcon/shared.h>

#include <map>
#include <deque>
#include <set>

namespace Falcon{

class EventCourier::Private
{
public:
   class CBData
   {
   public:
      int64 m_group;

      CBData()
      {}

      CBData(const Item& data, const String& message=""):
         m_group(0),
         m_lock(0),
         m_message(message)
      {
         if(data.isUser())
         {
            m_lock = Engine::instance()->collector()->lock(data);
         }
      }

      ~CBData()
      {
         if (m_lock != 0)
         {
            m_lock->dispose();
         }
      }

      CBData(const CBData& other):
         m_group(other.m_group),
         m_message(other.m_message)
      {
         if( other.m_lock != 0 )
         {
            m_lock = Engine::instance()->collector()->lock(other.m_lock->item());
         }
         else {
            m_lock = 0;
         }
      }

      const Item& data() const { return m_lock == 0 ? NIL : m_lock->item(); }
      void data( const Item& dt ) {
         if( m_lock == 0 )
         {
            Engine::instance()->collector()->lock(Item());
         }
         m_lock->item().copyFromLocal(dt);
      }

      const String& message() const { return m_message; }

      void setSummon( const Item& target, const String& message )
      {
         data(target);
         m_message = message;
      }

   private:

      GCLock* m_lock;
      String m_message;
      const Item NIL;
   };

   Mutex m_mtxSubs;
   typedef std::set<int64> EventSet;
   typedef std::map<int64, EventSet > GroupMap;
   GroupMap m_groups;
   typedef std::map<int,CBData> CBMap;
   CBMap m_callbacks;

   typedef std::deque<Token*> MessageList;
   Mutex m_mtxMessages;
   MessageList m_messages;
   uint32 m_messagesVersion;

   Mutex m_mtxPool;
   MessageList m_pool;
   length_t m_poolSize;

   Private():
      m_poolSize(EventCourier::DEFAULT_POOL_SIZE)
   {
      m_messagesVersion = 0;
   }

   ~Private()
   {}


   void sendToken( Token* tk )
   {
      tk->incref();
      this->m_mtxMessages.lock();
      this->m_messages.push_back(tk);
      this->m_messagesVersion++;
      this->m_mtxMessages.unlock();
   }

   Token* popToken()
   {
      Token* tk = 0;

      this->m_mtxMessages.lock();
      if( ! this->m_messages.empty() )
      {
         tk = this->m_messages.back();
         this->m_messages.pop_back();
         this->m_messagesVersion++;
      }
      this->m_mtxMessages.unlock();

      return tk;
   }

   void markTraveling( uint32 mark )
   {
      this->m_mtxMessages.lock();
      uint32 curVersion;
      do {
         curVersion = m_messagesVersion;
         MessageList::iterator iter = this->m_messages.begin();
         while( iter != this->m_messages.end() && curVersion == m_messagesVersion )
         {
            Token* tk = *iter;
            ItemArray& theArr = tk->params();

            this->m_mtxMessages.unlock();
            theArr.gcMark(mark);
            this->m_mtxMessages.lock();

            ++iter;
         }
      }
      while( curVersion != m_messagesVersion );

      this->m_mtxMessages.unlock();

   }
};

EventCourier::EventCourier()
{
   m_mark = 0;
   m_lastGroupID = 0;
   _p = new Private;
   m_throwOnUnhandled = true;
}

EventCourier::~EventCourier()
{
   delete _p;
   if( m_defaultHanlderLock != 0 )
   {
      m_defaultHanlderLock->dispose();
   }
}

/*
void EventCourier::addToGroup( int64 groupID, int64 eventID)
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks[eventID].m_group = groupID;
   _p->m_groups[groupID].insert(eventID);
   _p->m_mtxSubs.unlock();
}

void EventCourier::removeFromGroup( int64 groupID, int64 eventID)
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks[eventID].m_group = 0;
   _p->m_groups[groupID].erase(eventID);
   _p->m_mtxSubs.unlock();
}
*/

void EventCourier::setCallback( int64 id, const Item& callback, const String& msg )
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks[id].setSummon(callback, msg);
   _p->m_mtxSubs.unlock();
}

void EventCourier::clearCallback( int64 id )
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks.erase(id);
   _p->m_mtxSubs.unlock();
}


void EventCourier::prepareContext( VMContext* ctx )
{
   static const ClassEventCourier* cls =
            static_cast<ClassEventCourier*>(Engine::instance()->stdHandlers()->eventCourierClass());

   Item self(cls, this);
   self.methodize(cls->waitFunction());
   ctx->callItem(self);
}

void EventCourier::terminate()
{
}


Shared* EventCourier::eventPosted() const
{
   if ( m_sharedPosted == 0)
   {
      throw FALCON_SIGN_XERROR( CodeError, e_state, .extra("Not yet assigned to a context") );
   }

   return m_sharedPosted;
}


void EventCourier::subscribe( int64 eventID, const Item& handler )
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks[eventID].data(handler);
   _p->m_mtxSubs.unlock();
}

void EventCourier::subscribe( int64 eventID, const Item& handler, const String& message )
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks[eventID].setSummon(handler, message);
   _p->m_mtxSubs.unlock();
}

void EventCourier::unsubscribe( int64 eventID )
{
   _p->m_mtxSubs.lock();
   _p->m_callbacks.erase(eventID);
   _p->m_mtxSubs.unlock();
}

length_t EventCourier::poolSize() const
{
   _p->m_mtxPool.lock();
   length_t res = _p->m_poolSize;
   _p->m_mtxPool.unlock();

   return res;
}

void EventCourier::poolSize( length_t pz )
{
   _p->m_mtxPool.lock();
   _p->m_poolSize = pz;
   while( _p->m_pool.size() > pz )
   {
      delete _p->m_pool.back();
      _p->m_pool.pop_back();
   }
   _p->m_mtxPool.unlock();
}


const Item* EventCourier::getHandler( int64 eventID, String& message ) const
{
   const Item* ret = 0;

   _p->m_mtxSubs.lock();
   Private::CBMap::const_iterator iter = _p->m_callbacks.find(eventID);
   if( iter != _p->m_callbacks.end() )
   {
      const Private::CBData& data = iter->second;
      message = data.message();
      ret = &data.data();
   }
   _p->m_mtxSubs.unlock();

   return ret;
}


EventCourier::Token* EventCourier::allocToken()
{
   Token* tk = 0;
   _p->m_mtxPool.lock();
   if( _p->m_pool.empty() )
   {
      _p->m_mtxPool.unlock();
      tk = new Token(this);
   }
   else {
      tk = _p->m_pool.back();
      _p->m_pool.pop_back();
      _p->m_mtxPool.unlock();
   }

   return tk;
}

void EventCourier::release( Token* token )
{
   _p->m_mtxPool.lock();
   if( _p->m_pool.size() >= _p->m_poolSize )
   {
      _p->m_mtxPool.unlock();
      delete token;
   }
   else {
      _p->m_pool.push_back(token);
      _p->m_mtxPool.unlock();
   }
}


const Item* EventCourier::getDefaultHandler() const
{
   const Item* result = 0;

   m_mtxDflt.lock();
   if( m_defaultHanlderLock != 0 )
   {
      result = m_defaultHanlderLock->itemPtr();
   }
   m_mtxDflt.unlock();

   return result;
}


void EventCourier::setDefaultHandler( const Item& df )
{
   m_mtxDflt.lock();
   if( m_defaultHanlderLock == 0 )
   {
      m_defaultHanlderLock = Engine::instance()->collector()->lock(df);
   }
   else {
      m_defaultHanlderLock->item().copyFromLocal( df );
   }
   m_mtxDflt.unlock();
}


void EventCourier::sendEvent( int64 id, Item* params, int pcount )
{
   Token* tk = allocToken();
   tk->prepare(id, params, pcount);
   _p->sendToken(tk);
   tk->decref();
}

EventCourier::Token* EventCourier::sendEvent( int64 id, Item* params, int pcount, Event* evt )
{
   Token* tk = allocToken();
   tk->prepare(id, params, pcount, evt);
   _p->sendToken(tk);
   return tk;
}

EventCourier::Token* EventCourier::sendEvent( int64 id, Item* params, int pcount, Shared* evt )
{
   Token* tk = allocToken();
   tk->prepare(id, params, pcount, evt);
   _p->sendToken(tk);
   return tk;
}

void EventCourier::sendEvent( int64 id, Item* params, int pcount, Callback* cbt )
{
   Token* tk = allocToken();
   tk->prepare(id, params, pcount, cbt);
   _p->sendToken(tk);
   tk->decref();
}

EventCourier::Token* EventCourier::popEvent() throw()
{
   return _p->popToken();
}


void EventCourier::onUnhandled( Token* tk, VMContext* )
{
   tk->unhandled();

   if (throwOnUnhandled())
   {
      throw FALCON_SIGN_XERROR( CodeError, e_msg_unhandled, .extra(String("").N(tk->eventID())) );
   }
}


void EventCourier::gcMark(uint32 mark)
{
   if( m_mark != mark )
   {
      m_mark = mark;
      _p->markTraveling(mark);
   }
}

//=============================================================================
// Token class
//

EventCourier::Token::Token(EventCourier* owner):
         m_owner(owner)
{
   m_mode = mode_none;
   m_error = 0;
   m_gcLockResult = 0;
}


EventCourier::Token::~Token()
{
   clear();
}

bool EventCourier::Token::wait(Item& result, int64 to)
{
   if( m_mode != mode_event )
   {
      throw FALCON_SIGN_XERROR( CodeError, e_setup, .extra("Not an event-waitable EventCourier::Token") );
   }

   bool complete = m_completion.event->wait(to);
   if( complete )
   {
      if( m_error != 0 )
      {
         throw m_error;
      }
      result.copyFromRemote( m_gcLockResult->item() );
   }

   return complete;
}


const Item& EventCourier::Token::result() const
{
   if( m_gcLockResult != 0 )
   {
      return m_gcLockResult->item();
   }
   return m_NIL;
}


void EventCourier::Token::incref()
{
   atomicInc(m_refcount);
}

void EventCourier::Token::decref()
{
   if (atomicDec(m_refcount) == 0 )
   {
      dispose();
   }
}

void EventCourier::Token::prepare( int64 id, Item* params, int32 pcount, Event* evt)
{
   m_mode = mode_event;
   m_completion.event = evt;
   m_evtID = id;

   fillParams( params, pcount );
}

void EventCourier::Token::prepare( int64 id, Item* params, int32 pcount, Shared* sh)
{
   m_mode = mode_shared;
   m_completion.shared = sh;
   m_evtID = id;

   fillParams( params, pcount );
}

void EventCourier::Token::prepare( int64 id, Item* params, int32 pcount, Callback* cb)
{
   m_mode = mode_cb;
   m_completion.cb = cb;
   m_evtID = id;

   fillParams( params, pcount );
}

void EventCourier::Token::prepare( int64 id, Item* params, int32 pcount )
{
   m_mode = mode_none;
   m_evtID = id;
   fillParams( params, pcount );
}

void EventCourier::Token::fillParams( Item* params, int32 pcount )
{
   m_params.resize(pcount);
   for( int i= 0; i < pcount; ++i )
   {
      m_params[i] = params[i];
   }
}


void EventCourier::Token::completed( const Item& result )
{

   switch( m_mode )
   {
   case mode_cb:
      (*m_completion.cb)(m_evtID, result);
      break;

   case mode_event:
      m_gcLockResult = Engine::instance()->collector()->lock(result);
      m_completion.event->set();
      break;

   case mode_shared:
      m_gcLockResult = Engine::instance()->collector()->lock(result);
      m_completion.shared->signal();
      break;

   case mode_none:
      /* Nothing to do */
      break;

   }
}

void EventCourier::Token::aborted( Error* e )
{

   switch( m_mode )
   {
   case mode_cb:
      (*m_completion.cb)(m_evtID, Item(), e);
      break;

   case mode_event:
      m_error = e;
      e->incref();
      m_completion.event->set();
      break;

   case mode_shared:
      m_error = e;
      e->incref();
      m_completion.shared->signal();
      break;

   case mode_none:
      /* Nothing to do */
      break;
   }
}


void EventCourier::Token::unhandled()
{
   // for now, just mark as completed with nil
   completed(Item());
}


void EventCourier::Token::gcMark( uint32 md )
{
   if( m_mark != md )
   {
      m_mark = md;
      m_params.gcMark(md);
   }
}


void EventCourier::Token::dispose()
{
   clear();
   m_owner->release(this);
}


void EventCourier::Token::clear()
{

   if( m_gcLockResult != 0 )
   {
      m_gcLockResult->dispose();
      m_gcLockResult = 0;
   }

   if ( m_error != 0 )
   {
      m_error->decref();
      m_error = 0;
   }
}

}

/* end of eventcourier */
