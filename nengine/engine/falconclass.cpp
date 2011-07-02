/*
   FALCON - The Falcon Programming Language.
   FILE: falconclass.cpp

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>
#include <falcon/inheritance.h>
#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/falconstate.h>
#include <falcon/expression.h>
#include <falcon/operanderror.h>
#include <falcon/optoken.h>
#include <falcon/vm.h>
#include <falcon/ov_names.h>

#include <map>
#include <list>


#define OVERRIDE_OP_NEG_ID       0

#define OVERRIDE_OP_ADD_ID       1
#define OVERRIDE_OP_SUB_ID       2
#define OVERRIDE_OP_MUL_ID       3
#define OVERRIDE_OP_DIV_ID       4
#define OVERRIDE_OP_MOD_ID       5
#define OVERRIDE_OP_POW_ID       6

#define OVERRIDE_OP_AADD_ID      7
#define OVERRIDE_OP_ASUB_ID      8
#define OVERRIDE_OP_AMUL_ID      9
#define OVERRIDE_OP_ADIV_ID      10
#define OVERRIDE_OP_AMOD_ID      11
#define OVERRIDE_OP_APOW_ID      12

#define OVERRIDE_OP_INC_ID       13
#define OVERRIDE_OP_DEC_ID       14
#define OVERRIDE_OP_INCPOST_ID   15
#define OVERRIDE_OP_DECPOST_ID   16

#define OVERRIDE_OP_CALL_ID      17

#define OVERRIDE_OP_GETINDEX_ID  18
#define OVERRIDE_OP_SETINDEX_ID  19
#define OVERRIDE_OP_GETPROP_ID   20
#define OVERRIDE_OP_SETPROP_ID   21

#define OVERRIDE_OP_COMPARE_ID   22
#define OVERRIDE_OP_ISTRUE_ID    23
#define OVERRIDE_OP_IN_ID        24
#define OVERRIDE_OP_PROVIDES_ID  25
#define OVERRIDE_OP_TOSTRING_ID  27

#define OVERRIDE_OP_COUNT_ID  27


#if OVERRIDE_OP_COUNT_ID != OVERRIDE_OP_COUNT
#error "Count of overrides was not the same of override name count"
#endif

namespace Falcon
{


class FalconClass::Private
{
public:

   typedef std::map<String, Property> MemberMap;
   MemberMap m_members;

   typedef std::list<Inheritance*> ParentList;
   ParentList m_inherit;

   typedef std::list<FalconState*> StateList;
   StateList m_states;

   ItemArray m_propDefaults;

   Private()
   {}

   ~Private()
   {
      ParentList::iterator pi = m_inherit.begin();
      while( pi != m_inherit.end() )
      {
         delete *pi;
         ++pi;
      }

      StateList::iterator si = m_states.begin();
      while( si != m_states.end() )
      {
         delete *si;
         ++si;
      }

   }
};


FalconClass::Property::~Property()
{
}

//=====================================================================
// The class
//

FalconClass::FalconClass( const String& name ):
   Class("Object" , FLC_CLASS_ID_OBJECT ),
   m_fc_name(name),
   m_shouldMark(false),
   m_init(0)
{
   _p = new Private;
   m_overrides = new Function*[OVERRIDE_OP_COUNT_ID];
}


FalconClass::~FalconClass()
{
   delete[] m_overrides;
   delete _p;
}


FalconInstance* FalconClass::createInstance() const
{
   // we just need to copy the defaults.
   FalconInstance* inst = new FalconInstance(this);
   inst->data().merge(_p->m_propDefaults);

   // someone else will initialize non-defaultable items.
   return inst;
}


bool FalconClass::addProperty( const String& name, const Item& initValue )
{
   Private::MemberMap& members = _p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   _p->m_members.insert( std::make_pair(name, Property( _p->m_propDefaults.length() ) ) );
   // the add the init value in the value lists.
   _p->m_propDefaults.append( initValue );

   // is this thing deep? -- if it is so, we should mark it
   if( initValue.isDeep() )
   {
      m_shouldMark = true;
   }
   
   return true;
}

    
bool FalconClass::addProperty( const String& name )
{
     Private::MemberMap& members = _p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   _p->m_members.insert( std::make_pair(name, Property( _p->m_propDefaults.length() ) ) );
   // the add the init value in the value lists.
   _p->m_propDefaults.append( Item() );

   return true;
}

   
bool FalconClass::addMethod( Function* mth )
{
   Private::MemberMap& members = _p->m_members;

   const String& name = mth->name();
   
   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   _p->m_members.insert( std::make_pair( name, Property( mth ) ));

   // see if the method is an override.
   if( mth->name() == OVERRIDE_OP_NEG ) m_overrides[OVERRIDE_OP_NEG_ID] = mth;
   
   else if( mth->name() == OVERRIDE_OP_ADD ) m_overrides[OVERRIDE_OP_ADD_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_SUB ) m_overrides[OVERRIDE_OP_SUB_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_MUL ) m_overrides[OVERRIDE_OP_MUL_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_DIV ) m_overrides[OVERRIDE_OP_DIV_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_MOD ) m_overrides[OVERRIDE_OP_MOD_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_POW ) m_overrides[OVERRIDE_OP_POW_ID] = mth;

   else if( mth->name() == OVERRIDE_OP_AADD ) m_overrides[OVERRIDE_OP_AADD_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_ASUB ) m_overrides[OVERRIDE_OP_ASUB_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_AMUL ) m_overrides[OVERRIDE_OP_AMUL_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_ADIV ) m_overrides[OVERRIDE_OP_ADIV_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_AMOD ) m_overrides[OVERRIDE_OP_AMOD_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_APOW ) m_overrides[OVERRIDE_OP_APOW_ID] = mth;

   else if( mth->name() == OVERRIDE_OP_INC ) m_overrides[OVERRIDE_OP_INC_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_DEC ) m_overrides[OVERRIDE_OP_DEC_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_INCPOST ) m_overrides[OVERRIDE_OP_INCPOST_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_DECPOST ) m_overrides[OVERRIDE_OP_DECPOST_ID] = mth;

   else if( mth->name() == OVERRIDE_OP_CALL ) m_overrides[OVERRIDE_OP_CALL_ID] = mth;

   else if( mth->name() == OVERRIDE_OP_GETINDEX ) m_overrides[OVERRIDE_OP_GETINDEX_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_SETINDEX ) m_overrides[OVERRIDE_OP_SETINDEX_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_GETPROP ) m_overrides[OVERRIDE_OP_GETPROP_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_SETPROP ) m_overrides[OVERRIDE_OP_SETPROP_ID] = mth;

   else if( mth->name() == OVERRIDE_OP_COMPARE ) m_overrides[OVERRIDE_OP_COMPARE_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_ISTRUE ) m_overrides[OVERRIDE_OP_ISTRUE_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_IN ) m_overrides[OVERRIDE_OP_IN_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_PROVIDES ) m_overrides[OVERRIDE_OP_PROVIDES_ID] = mth;
   else if( mth->name() == OVERRIDE_OP_TOSTRING ) m_overrides[OVERRIDE_OP_TOSTRING_ID] = mth;

   return true;
}

  
bool FalconClass::addInit( Function* init )
{
   if( m_init == 0 )
   {
      m_init = init;
      return true;
   }

   return false;
}


bool FalconClass::addParent( Inheritance* inh )
{
   Private::MemberMap& members = _p->m_members;

   const String& name = inh->className();
   
   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   _p->m_members.insert( std::make_pair(name, Property( inh ) ) );

   return true;
}


bool FalconClass::getProperty( const String& name, Item& target ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.find( name );
   // first time around?
   if ( iter == members.end() )
   {
      return false;
   }

   // determine what we have found
   Property& prop = iter->second;
   switch( prop.m_type )
   {
      case Property::t_prop:
         target = _p->m_propDefaults[ prop.m_value.id ];
         break;

      case Property::t_func:
         target = prop.m_value.func;
         break;

      case Property::t_inh:
         //TODO
         break;

      case Property::t_state:
         //TODO
         break;
   }

   return true;
}


const FalconClass::Property* FalconClass::getProperty( const String& name ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.find( name );
   // first time around?
   if ( iter == members.end() )
   {
      return 0;
   }

   // determine what we have found
   return &iter->second;
}

void FalconClass::gcMark( uint32 mark ) const
{
   if ( m_shouldMark )
   {
      _p->m_propDefaults.gcMark( mark );
   }
}


bool FalconClass::addState( FalconState* state )
{
   Private::MemberMap& members = _p->m_members;

   const String& name = state->name();
   
   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   _p->m_members.insert( std::make_pair(name, Property( state ) ) );

   return true;
}


void FalconClass::serialize( DataWriter* ) const
{
   //TODO
}


void FalconClass::deserialize( DataReader* )
{
   //TODO
}


void FalconClass::enumeratePropertiesOnly( PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      if( iter->second.m_type == Property::t_prop)
      {
         if( ! cb( iter->first, ++iter == members.end() ) )
         {
            break;
         }
      }
      else
      {
         ++iter;
      }
   }
}

//========================================================================
// Class override.
//



void FalconClass::dispose( void* self ) const
{
   delete static_cast<FalconInstance*>(self);
}


void* FalconClass::clone( void* source ) const
{
   return static_cast<FalconInstance*>(source)->clone();
}


void FalconClass::serialize( DataWriter* stream, void* self ) const
{
   static_cast<FalconInstance*>(self)->serialize(stream);
}


void* FalconClass::deserialize( DataReader* stream ) const
{
   // TODO
   FalconInstance* fi = new FalconInstance;
   try
   {
      fi->deserialize(stream);
   }
   catch( ... )
   {
      delete fi;
      throw;
   }
   return fi;
}



//=========================================================
// Class management
//

void FalconClass::gcMark( void* self, uint32 mark ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   inst->gcMark( mark );
}


void FalconClass::enumerateProperties( void* , PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      const String& name = iter->first;
      if( ! cb( name, ++iter == members.end() ) )
      {
         break;
      }
   }
}


bool FalconClass::hasProperty( void*, const String& propName ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.find( propName );

   return members.end() != iter;
}


void FalconClass::describe( void* instance, String& target, int depth, int maxlen ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   
   class Descriptor: public PropertyEnumerator
   {
   public:
      Descriptor( FalconInstance* inst, String& forming, int d, int l ):
         m_inst(inst),
         m_target( forming ),
         m_depth( d ),
         m_maxlen( l )
      {}

      virtual bool operator()( const String& name, bool bLast )
      {
         Item theItem;
         m_inst->getMember( name, theItem );
         String temp;
         theItem.describe( temp, m_depth-1, m_maxlen );
         m_target += name + "=" + temp;
         if( ! bLast )
         {
            m_target += ", ";
         }
         return true;
      }

   private:
      FalconInstance* m_inst;
      String& m_target;
      int m_depth;
      int m_maxlen;
   };

   Descriptor rator( inst, target, depth, maxlen );

   target = "Instance of " + fc_name() +"{" ;
   enumerateProperties( instance, rator );
   target += "}";
}


//=========================================================
// Operators.
//

inline void FalconClass::override_unary( VMContext* ctx, void*, int op, const String& opName ) const
{
   Function* override = m_overrides[op];

   // TODO -- use pre-caching of the desired method
   if( override != 0 )
   {
      ctx->call( override, 0, ctx->topData(), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(opName) );
   }
}

inline void FalconClass::override_binary( VMContext* ctx, void*, int op, const String& opName ) const
{
   Function* override = m_overrides[op];
   
   if( override )
   {
      Item* first, *second;
      OpToken token( ctx, first, second );

      // 1 parameter == second; which will be popped away,
      // while first will be substituted with the return value of the function.
      ctx->call ( override, 1, *first, true );
      token.abandon();
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(opName) );
   }
}


void FalconClass::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   // we just need to copy the defaults.
   FalconInstance* inst = new FalconInstance(this);
   inst->data().merge(_p->m_propDefaults);

   // we have to invoke the init method, if any
   if( m_init != 0 )
   {
      Item self( FALCON_GC_STORE( coll, this, inst ) );
      // use isExpr so the A register will be popped in the stack
      ctx->call( m_init, pcount, self, true );
   }
   else
   {
      ctx->stackResult( pcount, FALCON_GC_STORE( coll, this, inst ) );
   }
}


void FalconClass::op_neg( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_NEG_ID, OVERRIDE_OP_NEG );
}


void FalconClass::op_add( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ADD_ID, OVERRIDE_OP_ADD );
}


void FalconClass::op_sub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_SUB_ID, OVERRIDE_OP_SUB );
}


void FalconClass::op_mul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MUL_ID, OVERRIDE_OP_MUL );
}


void FalconClass::op_div( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_DIV_ID, OVERRIDE_OP_DIV );
}

void FalconClass::op_mod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_MOD_ID, OVERRIDE_OP_MOD );
}


void FalconClass::op_pow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_POW_ID, OVERRIDE_OP_POW );
}


void FalconClass::op_aadd( VMContext* ctx, void* self) const
{
   override_binary( ctx, self, OVERRIDE_OP_AADD_ID, OVERRIDE_OP_AADD );
}


void FalconClass::op_asub( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_ASUB_ID, OVERRIDE_OP_ASUB );
}


void FalconClass::op_amul( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMUL_ID, OVERRIDE_OP_AMUL );
}


void FalconClass::op_adiv( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_DIV_ID, OVERRIDE_OP_DIV );
}


void FalconClass::op_amod( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_AMOD_ID, OVERRIDE_OP_AMOD );
}


void FalconClass::op_apow( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_APOW_ID, OVERRIDE_OP_APOW );
}


void FalconClass::op_inc( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INC_ID, OVERRIDE_OP_INC );
}


void FalconClass::op_dec( VMContext* ctx, void* self) const
{
   override_unary( ctx, self, OVERRIDE_OP_DEC_ID, OVERRIDE_OP_DEC );
}


void FalconClass::op_incpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_INCPOST_ID, OVERRIDE_OP_INCPOST );
}


void FalconClass::op_decpost( VMContext* ctx, void* self ) const
{
   override_unary( ctx, self, OVERRIDE_OP_DECPOST_ID, OVERRIDE_OP_DECPOST );
}


void FalconClass::op_getIndex( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_GETINDEX_ID, OVERRIDE_OP_GETINDEX );
}


void FalconClass::op_setIndex( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_SETINDEX_ID];

   if( override != 0 )
   {
      Item* first, *second, *third;
      OpToken token( ctx, first, second, third );

      // Two parameters (second and third) will be popped,
      //  and first will be turned in the result.
      ctx->call( override, 2, *first, true );
      token.abandon();
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_SETINDEX) );
   }
}


void FalconClass::op_getProperty( VMContext* ctx, void* self, const String& propName ) const
{   
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Function* override = m_overrides[OVERRIDE_OP_GETPROP_ID];

   if( override != 0 )
   {
      Item i_first = ctx->topData();
      // I prefer to go safe and push a new string here.
      ctx->pushData( (new String(propName))->garbage() );

      // use the instance we know, as first can be moved away.
      ctx->call( override, 1, i_first, true );
   }
   else
   {
      inst->getMember( propName, ctx->topData() );
   }
}


void FalconClass::op_setProperty( VMContext* ctx, void* self, const String& propName ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   Function* override = m_overrides[OVERRIDE_OP_SETPROP_ID];

   if( override != 0 )
   {
      Item* first, *second;
      OpToken token( ctx, first, second );
      Item i_data = *first;
      Item i_self = *second;

      ctx->pushData( (new String(propName))->garbage() );
      ctx->pushData( i_data );
      ctx->pushCode( &m_removeSelf );

      // Don't mangle the stack, we have to change it.
      ctx->call( override, 2, i_self, false );
      token.abandon();
   }
   else
   {
      inst->setProperty( propName, ctx->opcodeParam(1) );
      ctx->popData();
   }
}


void FalconClass::op_compare( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_COMPARE_ID, OVERRIDE_OP_SETINDEX );
}


void FalconClass::op_isTrue( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_ISTRUE_ID];

   if( override != 0 )
   {
      // use the instance we know, as first can be moved away.
      ctx->call( override, 0, ctx->topData(), true );
   }
   else
   {
      // instances are always true.
      ctx->topData().setBoolean(true);
   }
}


void FalconClass::op_in( VMContext* ctx, void* self ) const
{
   override_binary( ctx, self, OVERRIDE_OP_IN_ID, OVERRIDE_OP_IN );
}


void FalconClass::op_provides( VMContext* ctx, void* self, const String& propName ) const
{
   Function* override = m_overrides[OVERRIDE_OP_CALL_ID];

   if( override != 0  )
   {
      Item i_self = ctx->topData();
      ctx->pushData( (new String(propName))->garbage() );
      ctx->call( override, 1, i_self, true );
   }
   else
   {
      ctx->topData().setBoolean( hasProperty( self, propName ) );
   }
}


void FalconClass::op_call( VMContext* ctx, int32 paramCount, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_CALL_ID];

   if( override != 0  )
   {
      ctx->call( override, paramCount, ctx->topData(), true );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(OVERRIDE_OP_CALL) );
   }
}


void FalconClass::op_toString( VMContext* ctx, void* ) const
{
   Function* override = m_overrides[OVERRIDE_OP_TOSTRING_ID];

   if( override != 0 )
   {
      ctx->call( override, 0, ctx->topData(), true );
   }
   else
   {
      String* str = new String("Instance of ");
      str->append( name() );
      ctx->topData() = str;
   }
}

void FalconClass::RemoveSelf::apply_( const PStep*, VMContext* ctx)
{
   // use the return of the function as the thing to be put on top of the stack
   ctx->stackResult( 2, ctx->regA() );
}

}

/* end of falconclass.cpp */
