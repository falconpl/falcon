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
#include <falcon/ov_names.h>
#include <falcon/accesserror.h>
#include <falcon/operanderror.h>
#include <falcon/vmcontext.h>
#include <falcon/fassert.h>
#include <falcon/trace.h>
#include <falcon/inheritance.h>
#include <falcon/pcode.h>

#include <map>
#include <vector>
#include <cstring>

namespace Falcon
{

class HyperClass::Private
{
public:
   typedef std::map<String, Property> PropMap;
   PropMap m_props;

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
   Class(name),
   _p( new Private ),
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
   m_overrides = new Property*[OVERRIDE_OP_COUNT];
   memset( m_overrides, 0, OVERRIDE_OP_COUNT* sizeof( Property* ));
   addParent( new Inheritance( master->name(), master ) );
}


HyperClass::~HyperClass()
{
   delete[] m_overrides;
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
   if( _p->m_props.find(parent->name()) != _p->m_props.end() )
   {
      return false;
   }

   // The master class is added immediately, and has parent ID == 0
   if( m_nParents > 0 )
   {
      // ... and it must not appare in the inheritance properties.
      _p->m_props[parent->name()] = Property( parent, -m_nParents );
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
      _p->m_props[pname] = Property( cls, m_nParents );
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

inline bool HyperClass::get_override( void* self, int op, Class*& cls, void*& udata ) const
{
   Property* override = m_overrides[op];

   if( override != 0 && override->m_itemId >= 0 )
   {
      ItemArray& data = *static_cast<ItemArray*>(self);
      data[override->m_itemId].forceClassInst(cls, udata);
      return true;
   }

   return false;
}


void HyperClass::op_create( VMContext* ctx, int32 pcount ) const
{
   Collector* coll = Engine::instance()->collector();

   // prepare the space.
   ItemArray* mData = new ItemArray(m_nParents);
   mData->resize(m_nParents);
   
   // respect goingdeep protocol
   ctx->goingDeep();

   // If we have a constructor in the main class, then we need to create a
   // -- consistent frame (to find the parameters).
   if( m_constructor )
   {
      ctx->makeCallFrame( m_constructor, pcount, FALCON_GC_STORE( coll, this, mData ), false );

      // now we add a finalization step.
      ctx->pushCode( &m_finishCreateStep );

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
            ctx->pushCode( bottom->compiledExpr() );
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
      
      // now we add a finalization step.
      ctx->pushCode( &m_finishInvokeStep );

      // Add the master creation step.
      ctx->pushCode( &m_invokeMasterStep );

      // and add the creation of sub-steps.
      if( _p->m_parents.size() > 1 )
      {
         ctx->pushCode( &m_createEmptyNext );
         ctx->currentCode().m_seqId = _p->m_parents.size() - 2;
         // now invoke creation.
         _p->m_parents.back()->parent()->op_create( ctx, 0 );
      }
      // let the VM to get the next entry
   }
}


void HyperClass::op_neg( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override(  self, OVERRIDE_OP_NEG_ID, cls, udata ) )
   {
      cls->op_neg( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_NEG) );
   }
}


void HyperClass::op_add( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_ADD_ID, cls, udata ) )
   {
      cls->op_add( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ADD) );
   }
}


void HyperClass::op_sub( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_SUB_ID, cls, udata ) )
   {
      cls->op_sub( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SUB) );
   }
}


void HyperClass::op_mul( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_MUL_ID, cls, udata ) )
   {
      cls->op_mul( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_MUL) );
   }
}


void HyperClass::op_div( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_DIV_ID, cls, udata ) )
   {
      cls->op_div( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DIV) );
   }
}

void HyperClass::op_mod( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_MOD_ID, cls, udata ) )
   {
      cls->op_mod( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_MOD) );
   }
}


void HyperClass::op_pow( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_POW_ID, cls, udata ) )
   {
      cls->op_pow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_POW) );
   }
}


void HyperClass::op_aadd( VMContext* ctx, void* self) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_AADD_ID, cls, udata ) )
   {
      cls->op_aadd( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AADD) );
   }
}


void HyperClass::op_asub( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_ASUB_ID, cls, udata ) )
   {
      cls->op_asub( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_ASUB) );
   }
}


void HyperClass::op_amul( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_AMUL_ID, cls, udata ) )
   {
      cls->op_amul( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AMUL) );
   }
}


void HyperClass::op_adiv( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_DIV_ID, cls, udata ) )
   {
      cls->op_adiv( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DIV) );
   }
}


void HyperClass::op_amod( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_AMOD_ID, cls, udata ) )
   {
      cls->op_amod( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam( e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_AMOD) );
   }
}


void HyperClass::op_apow( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_APOW_ID, cls, udata ) )
   {
      cls->op_apow( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_APOW) );
   }
}


void HyperClass::op_inc( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_INC_ID, cls, udata ) )
   {
      cls->op_inc( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_INC) );
   }
}


void HyperClass::op_dec( VMContext* ctx, void* self) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_DEC_ID, cls, udata ) )
   {
      cls->op_dec( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DEC) );
   }
}


void HyperClass::op_incpost( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_INCPOST_ID, cls, udata ) )
   {
      cls->op_incpost( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_INCPOST) );
   }
}


void HyperClass::op_decpost( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_DECPOST_ID, cls, udata ) )
   {
      cls->op_decpost( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_DECPOST) );
   }
}


void HyperClass::op_getIndex( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_GETINDEX_ID, cls, udata ) )
   {
      cls->op_getIndex( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_GETINDEX) );
   }
}


void HyperClass::op_setIndex( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_SETINDEX_ID, cls, udata ) )
   {
      cls->op_setIndex( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_SETINDEX) );
   }
}


void HyperClass::op_getProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_GETPROP_ID, cls, udata ) )
   {
      cls->op_getProperty( ctx, udata, propName );
   }
   else
   {
      Private::PropMap::const_iterator iter = _p->m_props.find( propName );
      if( iter != _p->m_props.end() )
      {
         const Property& prop = iter->second;
         ItemArray* ia = static_cast<ItemArray*>(self);

         // if < 0 it's a class.
         if( prop.m_itemId < 0 )
         {
            // so, turn the thing in the "self" of the class.
            ctx->topData() = ia->at(-prop.m_itemId);
         }
         else
         {
            Class* cls;
            void* udata;
            ia->at(prop.m_itemId).forceClassInst( cls, udata );
            cls->op_getProperty( ctx, udata, propName );
         }
      }
      else
      {
         throw new AccessError( ErrorParam(e_prop_acc, __LINE__, SRC ).extra(propName) );
      }
   }
}


void HyperClass::op_setProperty( VMContext* ctx, void* self, const String& propName ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_SETPROP_ID, cls, udata ) )
   {
      cls->op_setProperty( ctx, udata, propName );
   }
   else
   {
      Private::PropMap::const_iterator iter = _p->m_props.find( propName );
      if( iter != _p->m_props.end() )
      {
         const Property& prop = iter->second;
         ItemArray* ia = static_cast<ItemArray*>(self);

         // if < 0 it's a class.
         if( prop.m_itemId < 0 )
         {
            // you can't overwrite a base class.
            throw new AccessError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(propName) );
         }
         else
         {
            Class* cls;
            void* udata;
            ia->at(prop.m_itemId).forceClassInst( cls, udata );
            cls->op_setProperty( ctx, udata, propName );
         }
      }
      else
      {
         throw new AccessError( ErrorParam(e_prop_acc, __LINE__, SRC ).extra(propName) );
      }
   }
}


void HyperClass::op_compare( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_COMPARE_ID, cls, udata ) )
   {
      cls->op_compare( ctx, udata );
   }
   else
   {
      // we don't need the self object.
      ctx->popData();
      const Item& crand = ctx->topData();
      if( crand.type() == typeID() )
      {
         // we're all object. Order by ptr.
         ctx->topData() = (int64)(self > crand.asInst() ? 1 : (self < crand.asInst() ? -1 : 0));
      }
      else
      {
         // order by type
         ctx->topData() = (int64)( typeID() - crand.type() );
      }
   }
}


void HyperClass::op_isTrue( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_ISTRUE_ID, cls, udata ) )
   {
      cls->op_isTrue( ctx, udata );
   }
   else
   {
      // objects are always true.
      ctx->topData() = true;
   }
}


void HyperClass::op_in( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_IN_ID, cls, udata ) )
   {
      cls->op_in( ctx, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_IN) );
   }
}


void HyperClass::op_provides( VMContext* ctx, void* self, const String& propName ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_PROVIDES_ID, cls, udata ) )
   {
      cls->op_provides( ctx, udata, propName );
   }
   else
   {
      ctx->topData().setBoolean( hasProperty( self, propName ) );
   }
}


void HyperClass::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_CALL_ID, cls, udata ) )
   {
      cls->op_call( ctx, paramCount, udata );
   }
   else
   {
      throw new OperandError( ErrorParam(e_invop, __LINE__, SRC ).extra(OVERRIDE_OP_CALL) );
   }
}


void HyperClass::op_toString( VMContext* ctx, void* self ) const
{
   Class* cls;
   void* udata;

   if( get_override( self, OVERRIDE_OP_CALL_ID, cls, udata ) )
   {
      cls->op_toString( ctx, udata );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = str;
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
      ctx->pushCode( inh->compiledExpr() );
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
   fassert( pstep_params[0].type() == FLC_ITEM_USER );
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
   ctx->stackResult( pcount+3, Item( h, inst ) );
   // declare the data as in need of collection.
   ctx->topData().garbage();

}


void HyperClass::InvokeMasterStep::apply_(const PStep* ps, VMContext* ctx )
{
   Item* pstep_params = ctx->opcodeParams(2);
   fassert( pstep_params[0].isUser() );
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
   fassert( pstep_params[0].isUser() );
   fassert( pstep_params[1].isInteger() );

   // get self and count
   ItemArray* inst = static_cast<ItemArray*>(pstep_params[0].asInst());

   // save the topmost item in our instance arrays.
   register CodeFrame& cf = ctx->currentCode();
   inst->at(cf.m_seqId) = ctx->topData();
   ctx->popData();

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
