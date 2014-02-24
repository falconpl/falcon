/*
   FALCON - The Falcon Programming Language.
   FILE: hyperclass.cpp

   Class holding more user-type classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Jul 2011 11:56:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/hyperclass.cpp"

#include <falcon/trace.h>
#include <falcon/stdsteps.h>
#include <falcon/hyperclass.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>

#include <falcon/psteps/exprinherit.h>
#include <falcon/psteps/exprparentship.h>
#include <falcon/vmcontext.h>
#include <falcon/fassert.h>
#include <falcon/falconclass.h>
#include <falcon/module.h>

#include "classes/classmulti_private.h"

#include <map>
#include <vector>
#include <cstring>

namespace Falcon
{


HyperClass::HyperClass( FalconClass* master ):
   ClassMulti(master->name()),
   m_constructor(0),
   m_master( master ),
   m_nParents(0),
   m_ownParentship( true ),
   m_initParentsStep( this ),
   m_InitMasterExprStep( this )
{
   _p_base = new Private_base;
   addParentProperties( master );
   m_nParents++;
   m_parentship = new ExprParentship();
   m_category = e_c_hyperclass;
}


HyperClass::HyperClass( const String& name ):
   ClassMulti(name),
   m_constructor(0),
   m_master( 0 ),
   m_nParents(0),
   m_ownParentship( true ),
   m_initParentsStep(this),
   m_InitMasterExprStep( this )
{
   _p_base = new Private_base;
   m_master = new FalconClass("#master$" + name);
   m_nParents++;

   // we'd hardly create a hyperclass not to store at least a foreign class.
   m_parentship = new ExprParentship();
   m_parentship->setParent( &m_master->makeConstructor()->syntree() );
   m_category = e_c_hyperclass;
}


HyperClass::~HyperClass()
{
   delete m_master;
   if( m_ownParentship )
   {
      delete m_parentship;
   }
   delete _p_base;
}

bool HyperClass::addProperty( const String& name, const Item& initValue )
{
   return m_master->addProperty( name, initValue );
}

bool HyperClass::addProperty( const String& name, Expression* initExpr )
{
   return m_master->addProperty( name, initExpr );
}

bool HyperClass::addProperty( const String& name )
{
   return m_master->addProperty( name );
}

bool HyperClass::addMethod( Function* mth )
{
   return m_master->addMethod( mth );
}


bool HyperClass::addParent( Class* cls )
{
   ClassMulti::Private_base::PropMap& props = _p_base->m_props;
   // Is the class name shaded?
   if( props.find(cls->name()) != props.end() )
   {
      return false;
   }

   // ... and it must not appare in the inheritance properties.
   props[cls->name()] = ClassMulti::Property( cls, -m_nParents );

   addParentProperties( cls );
   m_nParents++;
   m_parentship->append( new ExprInherit( cls ) );
   return true;
}


void HyperClass::setParentship( ExprParentship* ps, bool bOwn )
{
   if( m_ownParentship )
   {
      delete m_parentship;
   }

   m_parentship = ps;
   m_ownParentship = bOwn;
   ClassMulti::Private_base::PropMap& props = _p_base->m_props;

   // adds parents first to last.
   for( int i = 0; i < m_parentship->arity(); ++i )
   {
      ExprInherit* inh = static_cast<ExprInherit*>(m_parentship->nth(i));
      Class* cls = inh->base();

      // ... Always override names of parent classes.
      props[cls->name()] = ClassMulti::Property( cls, -m_nParents );
      addParentProperties( cls );
      m_nParents++;
   }
}


Class* HyperClass::getParent( const String& name ) const
{
   for( int i = 0; i < m_parentship->arity(); ++i )
   {
      ExprInherit* inh = static_cast<ExprInherit*>(m_parentship->nth(i));
      if ( inh->base()->name() == name ) {
         return inh->base();
      }
   }

   return 0;
}

void HyperClass::addParentProperties( Class* cls )
{
   TRACE2("HyperClass::addParentProperties(%s)", cls->name().c_ize() );

   class PE: public PropertyEnumerator
   {
   public:
      PE( Class* cls, HyperClass* owner ):
         m_cls(cls),
         m_owner(owner)
      {}

      virtual bool operator()( const String& pname )
      {
         // ignore properties representing parent classes.
         if( m_cls->getParent( pname ) == 0 )
         {
            TRACE2("HyperClass::addParentProperties(%s) -- adding %s", m_cls->name().c_ize(), pname.c_ize());
            m_owner->addParentProperty( m_cls, pname );
         }

         return true;
      }

   private:
      Class* m_cls;
      HyperClass* m_owner;
   };

   PE cb( cls, this );

   cls->enumerateProperties( 0, cb );
}


void HyperClass::addParentProperty( Class* cls, const String& pname )
{
   if( _p_base->m_props.find( pname ) == _p_base->m_props.end() )
   {
      Property &p = (_p_base->m_props[pname] = Property( cls, m_nParents ));
      checkAddOverride( pname, &p );
   }
   // else, the property is masked by an higher priority class.
}

void HyperClass::dispose( void* self ) const
{
   delete static_cast<ItemArray*>(self);
}


void* HyperClass::clone( void* source ) const
{
   return new ItemArray( *static_cast<ItemArray*>(source) );
}

void* HyperClass::createInstance() const
{
   TRACE1("HyperClass(%p,%s)::createInstance() for %d parents", this, name().c_ize(), m_nParents );

   ItemArray* ia = new ItemArray( m_nParents );
   ia->resize(m_nParents);

   // TODO: maybe we can posticipate this to init?
   for( int i = 0; i < m_nParents; ++i )
   {
      Class* cls = i==0 ? m_master :
         static_cast<ExprInherit*>( m_parentship->nth(i-1) )->base();

      TRACE1("HyperClass(%p,%s)::createInstance() for %s",
               this, name().c_ize(), cls->name().c_ize() );

      if( ! cls->isFlatInstance() ) {
         (*ia)[i] = FALCON_GC_STORE( cls, cls->createInstance() );
      }
      // else, the init step will fill the object.
   }

   // our entity will be put in the garbage as soon as we return.
   return ia;
}


bool HyperClass::isDerivedFrom( const Class* cls ) const
{
   // are we the required class -- the master is an overkill, should not be visible
   if ( cls == this || cls == m_master ) return true;

   // is the class a parent of one of our parents?
   for( int i = 0; i < m_parentship->arity(); ++i ) {
      Class* pcls = static_cast<ExprInherit*>( m_parentship->nth(i) )->base();

      if( cls->isDerivedFrom(pcls) ) {
         return true;
      }
   }

   return false;
}


void* HyperClass::getParentData( const Class* parent, void* data ) const
{
   // are we the searched parent?
   if( parent == this || parent == m_master )
   {
      // then the searched data is the given one.
      return data;
   }

   // else, search the parent data among our parents.
   // parent data is stored in an itemarray in data,
   // -- parent N data is at position N.
   ItemArray& ia = *static_cast<ItemArray*>(data);

   // is the class a parent of one of our parents?
   for( uint32 i = 0; i < ia.length(); ++i )
   {
      // get the nth parent from our parent vector
      // -- be sure to cover also flat instance classes.
      Class* pcls;
      void* pinst;
      ia[i].forceClassInst( pcls, pinst );

      // Seek parent data recursively
      void* dp = pcls->getParentData( parent, pinst );
      if( dp != 0 )
      {
         // it was either pinst or a fraction of it.
         return dp;
      }
   }

   // no luck.
   return 0;
}


void HyperClass::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      m_mark = mark;
      m_parentship->gcMark( mark );

      // shouldn't really care, but just in case...
      m_master->gcMark( mark );

      // Also mark the constructor and finally the module (if any).
      if( m_module != 0 ) m_module->gcMark(mark);
      if( m_constructor != 0 ) m_constructor->gcMark( mark );

      // finally all our parents
      for( int i = 0; i < m_parentship->arity(); ++i ) {
         Class* pcls = static_cast<ExprInherit*>( m_parentship->nth(i) )->base();
         pcls->gcMark( mark );
      }
   }
}


void HyperClass::gcMarkInstance( void* self, uint32 mark ) const
{
   static_cast<ItemArray*>(self)->gcMark( mark ); // which marks also the parent classes
}


void HyperClass::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   ClassMulti::Private_base::PropMap& props = _p_base->m_props;
   ClassMulti::Private_base::PropMap::const_iterator iter = props.begin();

   while( iter != props.end() )
   {
      const String& prop = iter->first;
      cb( prop );
      ++ iter;
   }
}


void HyperClass::enumeratePV( void* data, PVEnumerator& cb ) const
{
   ItemArray& ia = *static_cast<ItemArray*>(data);

   // is the class a parent of one of our parents?
   for( uint32 i = 0; i < ia.length(); ++i )
   {
      // get the nth parent from our parent vector
      // -- be sure to cover also flat instance classes.
      Class* pcls;
      void* pinst;
      ia[i].forceClassInst( pcls, pinst );

      pcls->enumeratePV(pinst, cb);
   }
}


bool HyperClass::hasProperty( void*, const String& prop ) const
{
   ClassMulti::Private_base::PropMap& props = _p_base->m_props;
   ClassMulti::Private_base::PropMap::const_iterator iter = props.find( prop );
   return iter != props.end();
}


Class* HyperClass::getParentAt( int pos ) const
{
   Class* pcls = static_cast<ExprInherit*>( m_parentship->nth(pos) )->base();
   return pcls;
}


void HyperClass::describe( void* instance, String& target, int depth, int maxlen ) const
{
   if( depth == 0 )
   {
      target = name() + "{...}";
      return;
   }

   String str = name() + "{";

   ItemArray& ia = *static_cast<ItemArray*>(instance);

   // is the class a parent of one of our parents?
   for( uint32 i = 0; i < ia.length(); ++i )
   {
      // get the nth parent from our parent vector
      // -- be sure to cover also flat instance classes.
      Class* pcls;
      void* pinst;
      ia[i].forceClassInst( pcls, pinst );
      if( i > 0 )
      {
         str += "; ";
      }

      String temp;
      pcls->describe( pinst, temp, depth-1, maxlen );
      str += temp;
   }
   str += "}";

   target  = str;
}

//=========================================================
// Operators.
//

bool HyperClass::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   //static StdSteps& st = *Engine::instance()->stdSteps();

   ItemArray& mData = *static_cast<ItemArray*>(instance);
   TRACE1("HyperClass(%p,%s)::op_init(ctx:%p, inst:%p, pcount: %d)",
            this, name().c_ize(), ctx, instance, pcount );

   // if we have a constructor, call it.
   if( m_constructor )
   {
      ctx->callInternal( m_constructor, pcount, Item( this, &mData ) );

      // we'll have to call also the master constructor. -- if so, repush the params.
      if( m_master->constructor() != 0 )
      {
         ctx->forwardParams(pcount);
      }
   }

   if( m_master->constructor() != 0 )
   {
      // good, we have a master class constructor.
      // we also know that the master class has no direct parentship -- we own it.
      // We finally know that the constructor is made to work with our class instead of FalconClass,
      // as it uses a generic self. accessor (it's Falcon code), so we can send our SELF, and not data[0]
      // where the master class data lies.
      //ctx->pushCode( &st.m_pop );
      ctx->callInternal( m_master->constructor(), pcount, ctx->opcodeParam(pcount) );
      ctx->pushCode(&m_InitMasterExprStep);
   }

   // we finally know that the master op_init went deep, as it has a ctor...
   ctx->pushData(Item(this,instance));
   ctx->stepIn( &m_initParentsStep );

   // stepInit is always to be called, prevent stack cleaning
   return m_constructor != 0 || m_master->constructor() != 0;
}



//========================================================
// Steps
//

void HyperClass::InitParentsStep::apply_(const PStep* ps, VMContext* ctx )
{
   const InitParentsStep* self = static_cast<const InitParentsStep*>(ps);
   CodeFrame& cf = ctx->currentCode();
   int& pid = cf.m_seqId;

   ItemArray& mData = *static_cast<ItemArray*>(ctx->opcodeParam(pid).asInst());
   int size = mData.length()-1;


   TRACE1("HyperClass(%p,%s)::InitParentsStep::apply_ %d/%d (depth %d)",
                  self->m_owner, self->m_owner->name().c_ize(), pid, size, (int) ctx->dataSize() );

   while( pid < size )
   {
      // ei wants the item before being called.
      ExprInherit* ei =
               static_cast<ExprInherit*>(self->m_owner->m_parentship->nth(pid));
      // data in mData goes 1..size
      ++pid;
      // ExprInherit wants the self object on top of the data stack.
      ctx->pushData( mData[pid] );
      if( ctx->stepInYield( ei, cf ) ) {
         return;
      }
   }

   // we're done.
   ctx->popCode();
   // pop also the extra data we pushed (An extra copy of self) at op_init
   ctx->popData(pid+1);
}

void HyperClass::InitMasterExprStep::apply_(const PStep* ps, VMContext* ctx )
{
   ItemArray& mData = *static_cast<ItemArray*>(ctx->currentFrame().m_self.asInst());
   FalconInstance* inst = static_cast<FalconInstance*>( mData[0].asInst() );
   FalconClass* fc = static_cast<FalconClass*>( mData[0].asClass() );

   (void) ps;

#ifndef NDEBUG
   const InitMasterExprStep* self = static_cast<const InitMasterExprStep*>(ps);

   TRACE1("HyperClass(%p,%s)::InitMasterExprStep::apply_ %d (depth %d)",
                  self->m_owner, self->m_owner->name().c_ize(), ctx->currentCode().m_seqId, (int) ctx->dataSize() );
#endif

   FalconClass::applyInitExpr(ctx, fc, inst);
}

}

/* end of hyperclass.cpp */
