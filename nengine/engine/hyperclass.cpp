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

#include <map>
#include <vector>

namespace Falcon
{

class HyperClass::Private
{
public:
   Private() {}
   ~Private() {}

   typedef std::map<String, Property> PropMap;
   PropMap m_props;

   typedef std::vector<Class*> ParentVector;
   ParentVector m_parents;
};


HyperClass::HyperClass( const String& name, Class* master ):
   Class(name),
   _p( new Private ),
   m_master( master ),
   m_nParents(0),
   m_initStep( this )
{
   m_overrides = new Property*[OVERRIDE_OP_COUNT];
   addParentProperties( master );
   _p->m_parents.push_back( master );
   m_nParents++;
}

HyperClass::~HyperClass()
{
   delete[] m_overrides;
   delete m_master;
   delete _p;
}

  
bool HyperClass::addParent( Class* cls )
{
   // Is the class name shaded?
   if( _p->m_props.find(cls->name()) !=  _p->m_props.end() )
   {
      return false;
   }
   _p->m_props[cls->name()] = Property( cls, -m_nParents );
   addParentProperties( cls );
   m_nParents++;
   _p->m_parents.push_back( cls );
   return true;
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
         m_owner->addParentProperty( m_cls, pname );
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


void HyperClass::gcMarkMyself( uint32 mark ) const
{
   Private::PropMap::iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      iter->second.m_provider->gcMarkMyself( mark );
      ++ iter;
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
   return _p->m_parents[pos];
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
   
   // push self
   ctx->pushData( FALCON_GC_STORE( coll, this, mData ) );
   // save the parameter counts
   ctx->pushData( pcount );

   // have the VM to call init.
   ctx->pushCode( &m_initStep );
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

// This is used to initialize this class calling the op_create of all the members.
HyperClass::PStepInit::PStepInit( HyperClass* o ):
   m_owner(o)
{
   apply = apply_;
}

void HyperClass::PStepInit::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepInit* istep = static_cast<const PStepInit*>(ps);
   HyperClass* owner = istep->m_owner;
   TRACE("HyperClass init step for class %s", owner->name().c_ize() );

   CodeFrame& frame = ctx->currentCode();
   ItemArray* pia;
   
   // not the first time around?
   if( frame.m_seqId > 0 )
   {
      // ...  we got things to save,
      // ... and we have our array at third-last place instach
      fassert( ctx->opcodeParam(-2).asClass() == owner );
      pia = static_cast<ItemArray*>( ctx->opcodeParam(-2).asInst() );
      pia->at(frame.m_seqId-1) = ctx->topData();
      // remove the data that we have saved.
      ctx->popData();
   }
   else
   {
      // Our array is second-last
      fassert( ctx->opcodeParam(-1).asClass() == owner );
      pia = static_cast<ItemArray*>( ctx->opcodeParam(-1).asInst() );
   }

   ItemArray& ia = *pia;
   int32 len = (int32) ia.length();

   // count of parameters are in the topmost part of the stack.
   fassert( ctx->opcodeParam(0).isInteger() );
   int32 pcount = (int32) ctx->opcodeParam(0).asInteger();

   // check that we're not leaving dirty things in the stack.
#ifndef NDEBUG
   size_t data_depth = (size_t) ctx->dataSize();
#endif

   // proceed to create new instances as long as needed.
   // repeat until some base class needs to go deep.
   while( frame.m_seqId < len )
   {
      int seqId = frame.m_seqId++;
      Class* cls = owner->getParentAt( seqId );
      // re-push all the parameters.
      int count = pcount-1;
      while( count >= 0 )
      {
         ctx->pushData( ctx->opcodeParam(count + 2) );
         --count;
      }
      
      cls->op_create( ctx, pcount );
      // going deep?
      if( &frame != &ctx->currentCode() )
      {
         // went deep! -- the data will be saved later.
         TRACE1("Class %s went deep, suspending.", cls->name().c_ize() );
         return;
      }

      // else, we can save our little data now.
      ia[seqId] = ctx->topData();
      ctx->popData();
      
      // at this point the depth of the stack must be the same.
#ifndef NDEBUG
      fassert( data_depth == (size_t) ctx->dataSize() );
#endif
   }
  
   // we're done
   ctx->popCode();
   // commit the data -- and remove the parameters.
   ctx->popData( pcount + 2 - 1);
   ctx->topData().setUser( owner, &ia );
   ctx->topData().garbage();
}

}

/* end of hyperclass.cpp */
