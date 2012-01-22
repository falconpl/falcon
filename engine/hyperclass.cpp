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

#include <falcon/hyperclass.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/fassert.h>
#include <falcon/trace.h>
#include <falcon/inheritance.h>

#include "multiclass_private.h"

#include <map>
#include <vector>
#include <cstring>

namespace Falcon
{

class HyperClass::Private: public MultiClass::Private_base
{
public:
   
   typedef std::vector<Inheritance*> ParentVector;
   ParentVector m_parents;

   Private() {}
   ~Private()
   {
      // we own our inheritances.
      ParentVector::iterator piter = m_parents.begin();
      while( m_parents.end() != piter )
      {
         delete *piter;
         ++piter;
      }
   }
};


HyperClass::HyperClass( const String& name, Class* master ):
   MultiClass(name),   
   m_master( master ),
   m_nParents(0),
   m_constructor(0),
      
   m_finishCreateStep(this),
   m_createMasterStep(this),
   m_parentCreatedStep(this),
   m_createParentStep(this),

   m_finishInvokeStep(this),
   m_invokeMasterStep(this),
   m_createEmptyNext(this)
{
   _p = new Private;
   _p_base = _p;
   addParent( new Inheritance( name, master ) );
}


HyperClass::~HyperClass()
{
   delete m_master;
   delete _p;
}

  
bool HyperClass::addParent( Inheritance* cls )
{
   Class* parent = cls->parent();
   
   // we accept only ready inheritances.
   if( parent == 0 )
   {
      return false;
   }

   // Is the class name shaded?
   if( _p->m_props.find(cls->className()) != _p->m_props.end() )
   {
      return false;
   }

   // The master class is added immediately, and has parent ID == 0
   if( m_nParents > 0 )
   {
      // ... and it must not appare in the inheritance properties.
      _p->m_props[cls->className()] = Property( parent, -m_nParents );
   }

   addParentProperties( parent );
   m_nParents++;
   _p->m_parents.push_back( cls );
   return true;
}

Class* HyperClass::getParent( const String& name ) const
{
   Private::ParentVector::const_iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      if( name == (*iter)->parent()->name() )
      {
         return (*iter)->parent();
      }

      ++iter;
   }
   
   return 0;
}

void HyperClass::addParentProperties( Class* cls )
{
   class PE: public PropertyEnumerator
   {
   public:
      PE( Class* cls, HyperClass* owner ):
         m_cls(cls),
         m_owner(owner)
      {}

      virtual bool operator()( const String& pname, bool )
      {
         // ignore properties representing parent classes.
         if( m_cls->getParent( pname ) == 0 )
         {
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
   if( _p->m_props.find( pname ) == _p->m_props.end() )
   {
      Property &p = (_p->m_props[pname] = Property( cls, m_nParents ));
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


void HyperClass::serialize( DataWriter*, void* ) const
{
   // TODO.
}


void* HyperClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}


bool HyperClass::isDerivedFrom( const Class* cls ) const
{
   // are we the required class?
   if ( cls == this ) return true;
   
   // is the class a parent of one of our parents?
   Private::ParentVector::const_iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      Inheritance* inh = *iter;
      // notice that isDerivedFrom returns true also if cls == parent.
      if( inh->parent() != 0 && inh->parent()->isDerivedFrom(cls) )
      {
         return true;
      }
      ++iter;
   }
   
   return false;
}


void* HyperClass::getParentData( Class* parent, void* data ) const
{
   // are we the searched parent?
   if( parent == this )
   {
      // then the searched data is the given one.
      return data;
   }
      
   // else, search the parent data among our parents.
   // parent data is stored in an itemarray in data, 
   // -- parent N data is at position N.
   ItemArray* ia = static_cast<ItemArray*>(data);
   Private::ParentVector::const_iterator iter = _p->m_parents.begin();
   length_t count = 0;
   // ... so we scan the parent vector...
   while( iter != _p->m_parents.end() )
   {
      Inheritance* inh = *iter;
      if( inh->parent() != 0 )
      {
         // ... and ask each parent to find the parent data in its data.
         void* dp = inh->parent()->getParentData( parent, ia->at(count).asInst() );
         if( dp != 0 )
         {
            return dp;
         }
      }
      ++iter;
      ++count;
   }
   
   // no luck.
   return 0;
}


void HyperClass::gcMarkMyself( uint32 mark )
{
   if( m_lastGCMark != mark )
   {
      m_lastGCMark = mark;

      Private::ParentVector::iterator iter = _p->m_parents.begin();
      while( iter != _p->m_parents.end() )
      {
         (*iter)->parent()->gcMarkMyself( mark );
         ++ iter;
      }
   }
}


void HyperClass::gcMark( void* self, uint32 mark ) const
{
   static_cast<ItemArray*>(self)->gcMark( mark );
}


void HyperClass::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   Private::PropMap::const_iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      const String& prop = iter->first;
      ++ iter;
      cb( prop, iter == _p->m_props.end() );
   }
}


void HyperClass::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Private::ParentVector::const_reverse_iterator iter = _p->m_parents.rbegin();
   while( iter != _p->m_parents.rend() )
   {
      (*iter)->parent()->enumeratePV( self, cb );
      ++ iter;
   }
}


bool HyperClass::hasProperty( void*, const String& prop ) const
{
   Private::PropMap::iterator iter = _p->m_props.find( prop );
   return iter != _p->m_props.end();
}


Class* HyperClass::getParentAt( int pos ) const
{
   return _p->m_parents[pos]->parent();
}


void HyperClass::describe( void* instance, String& target, int depth, int maxlen ) const
{
   String str;
   ItemArray& ia = *static_cast<ItemArray*>(instance);

   Class* cls;
   void* udata;
   ia[0].forceClassInst( cls, udata );

   m_master->describe( udata, str, depth, maxlen );
   target = str;

   Private::PropMap::const_iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      const Property& prop = iter->second;
      if ( prop.m_itemId < 0 )
      {
         str = "";
         ia[ -prop.m_itemId ].forceClassInst( cls, udata );
         cls->describe( udata, str, depth, maxlen );
         target += "; " + str;
      }
      ++iter;
   }
}

//=========================================================
// Operators.
//

void HyperClass::op_create( VMContext* ctx, int32 pcount ) const
{
   Collector* coll = Engine::instance()->collector();

   // prepare the space.
   ItemArray* mData = new ItemArray(m_nParents);
   mData->resize(m_nParents);
   
   // If we have a constructor in the main class, then we need to create a
   // -- consistent frame (to find the parameters).
   if( m_constructor )
   {
      ctx->makeCallFrame( m_constructor, pcount, FALCON_GC_STORE( coll, this, mData ) );

      // now we add a finalization step.
      ctx->pushCode( &m_finishCreateStep );

      ctx->pushData(Item()); // create a space used for op_create() unary constance
      
      // repeat the data as local for the master initialization.
      for( int i = 0; i < pcount; ++i )
      {
         ctx->pushData( *ctx->param(i) );
      }

      // Add the master creation step.
      ctx->pushCode( &m_createMasterStep );

      if( _p->m_parents.size() > 1 )
      {
         // invoke the first creation step.
         Inheritance* bottom = _p->m_parents.back();
         ctx->pushCode( &m_parentCreatedStep );
         ctx->currentCode().m_seqId = _p->m_parents.size() - 1;

         if( bottom->paramCount() > 0 )
         {
            // ask to create the class...
            ctx->pushCode( &m_createParentStep );
            ctx->currentCode().m_seqId = _p->m_parents.size() - 1;
            //... after having created the parameters.
            bottom->prepareOnContext( ctx );
         }
         else
         {
            // just invoke the creation of the class.
            bottom->parent()->op_create( ctx, 0 );
         }
      }
   }
   else
   {
      // We have no constructor -- this is probably a prototype.
      // all the subclasses must be invoked parameterless.

      // push the required data:
      ctx->pushData(FALCON_GC_STORE( coll, this, mData ));
      ctx->pushData( (int64) pcount );
      ctx->pushData(Item()); // create a space used for op_create() unary constance
      
      // now we add a finalization step.
      ctx->pushCode( &m_finishInvokeStep );

      // Add the master creation step.
      ctx->pushCode( &m_invokeMasterStep );

      // and add the creation of sub-steps.
      if( _p->m_parents.size() > 1 )
      {
         ctx->pushCode( &m_createEmptyNext );
         ctx->currentCode().m_seqId = _p->m_parents.size() - 1;
         // now invoke creation.
         _p->m_parents.back()->parent()->op_create( ctx, 0 );
      }
      // let the VM to get the next entry
   }
}



//========================================================
// Steps
//

void HyperClass::FinishCreateStep::apply_(const PStep*, VMContext* ctx )
{
   // we have to set the topmost data returned by create master step...
   // copy it by value
   Item self = ctx->currentFrame().m_self;
   static_cast<ItemArray*>(self.asInst())->at(0) = ctx->topData();
   // and then return the frame...
   ctx->returnFrame();
   // and push the self on top.
   ctx->pushData( self );
}


void HyperClass::CreateMasterStep::apply_(const PStep* ps, VMContext* ctx )
{
   // we're around just once.
   ctx->popCode();
   // just invoke the creation. HyperClass::op_create has already pushed the params.
   HyperClass* h = static_cast<const CreateMasterStep*>(ps)->m_owner;
   h->m_master->op_create( ctx, ctx->currentFrame().m_paramCount );
}


void HyperClass::ParentCreatedStep::apply_(const PStep* ps, VMContext* ctx )
{
   // Save the created parent.
   Item& self = ctx->currentFrame().m_self;
   register CodeFrame& cf = ctx->currentCode();
   static_cast<ItemArray*>(self.asInst())->at(cf.m_seqId) = ctx->topData();
   
   // and we don't need it anymore
   ctx->popData();
   
   // then see what's nest. Need to create more?
   if( -- cf.m_seqId == 0 )
   {
      // we're done -- parent[0] is the master class.
      ctx->popCode();
      return;
   }

   // get the required parent.
   HyperClass* h = static_cast<const ParentCreatedStep*>(ps)->m_owner;
   Inheritance* inh = h->_p->m_parents[cf.m_seqId];
   if( inh->paramCount() == 0 )
   {
      // just create.
      inh->parent()->op_create( ctx, 0 );
   }
   else
   {
      // save the needed steps.
      int seqId = cf.m_seqId;
      ctx->pushCode( &h->m_createParentStep );
      ctx->currentCode().m_seqId = seqId;
      inh->prepareOnContext( ctx );
   }
   // let the VM take care of the rest.
}


void HyperClass::CreateParentStep::apply_(const PStep* ps, VMContext* ctx )
{
   register CodeFrame& cf = ctx->currentCode();

   // get the required parent.
   HyperClass* h = static_cast<const CreateParentStep*>(ps)->m_owner;
   Inheritance* inh = h->_p->m_parents[cf.m_seqId];

   // remove ourselves, we just live once.
   ctx->popCode();

   // and invoke creation with the required parameters.
   inh->parent()->op_create( ctx, inh->paramCount() );
}



void HyperClass::FinishInvokeStep::apply_(const PStep* ps, VMContext* ctx )
{
   // get the data created by the master
   Item* pstep_params = ctx->opcodeParams(3);

   // get self
   fassert( pstep_params[0].type() >= FLC_ITEM_USER );
   ItemArray* inst = static_cast<ItemArray*>(pstep_params[0].asInst());
   // save the data created by the master class
   inst->at(0) = pstep_params[2];

   // get the param count.
   fassert( pstep_params[1].type() == FLC_ITEM_INT );
   int pcount = (int) pstep_params[1].asInteger();

   // we're not needed anymore
   ctx->popCode();
   
   //remove the params and publish self.
   HyperClass* h = static_cast<const FinishInvokeStep*>(ps)->m_owner;
   ctx->stackResult( pcount+1+3, Item( h, inst ) );
   // declare the data as in need of collection.

}


void HyperClass::InvokeMasterStep::apply_(const PStep* ps, VMContext* ctx )
{
   Item* pstep_params = ctx->opcodeParams(3);
   fassert( pstep_params[0].asClass() == static_cast<const InvokeMasterStep*>(ps)->m_owner );
   fassert( pstep_params[1].isInteger() );

   int pcount = (int) pstep_params[1].asInteger();

   // replicate the parameters.
   if( pcount > 0 )
   {
      // get the real parameters
      ctx->addSpace(pcount);
      Item* params = ctx->opcodeParams(2+pcount*2);
      Item* pdest = ctx->opcodeParams(pcount);
      Item* pterm = pdest + pcount;
      while( pdest < pterm )
      {
         *pdest = *params;
         ++pdest;
         ++params;
      }
   }

   // we're not needed anymore
   ctx->popCode();

   HyperClass* h = static_cast<const InvokeMasterStep*>(ps)->m_owner;
   h->m_master->op_create( ctx, pcount );
}

   
void HyperClass::CreateEmptyNext::apply_(const PStep* ps, VMContext* ctx )
{
   // we have the 2 pushed stuff plus our newly created item.
   Item* pstep_params = ctx->opcodeParams(3);
   fassert( pstep_params[0].asClass() == static_cast<const CreateEmptyNext*>(ps)->m_owner );
   fassert( pstep_params[1].isInteger() );

   // get self and count
   ItemArray* inst = static_cast<ItemArray*>(pstep_params[0].asInst());

   // save the topmost item in our instance arrays.
   register CodeFrame& cf = ctx->currentCode();
   inst->at(cf.m_seqId) = ctx->topData();
   // the topmost data will be recycled as op_create unary operator item

   // are we done?
   if( -- cf.m_seqId == 0 )
   {
      // let the VM to finalize us.
      ctx->popCode();
   }
   else
   {
      // invoke the next operator
      HyperClass* h = static_cast<const CreateEmptyNext*>(ps)->m_owner;
      h->_p->m_parents[cf.m_seqId]->parent()->op_create( ctx, 0 );
   }
}
   
}

/* end of hyperclass.cpp */
