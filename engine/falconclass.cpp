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

#undef SRC
#define SRC "engine/falconclass.cpp"

#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/falconinstance.h>
#include <falcon/inheritance.h>
#include <falcon/item.h>
#include <falcon/synfunc.h>
#include <falcon/function.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/falconstate.h>
#include <falcon/expression.h>
#include <falcon/operanderror.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/expression.h>
#include <falcon/ov_names.h>
#include <falcon/trace.h>

#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/stmtinit.h>

#include <map>
#include <list>
#include <vector>

namespace Falcon
{


class FalconClass::Private
{
public:

   typedef std::map<String, Property*> MemberMap;
   MemberMap m_members;

   typedef std::list<Inheritance*> ParentList;
   ParentList m_inherit;

   typedef std::list<FalconState*> StateList;
   StateList m_states;

   typedef std::vector<Property*> InitPropList;
   InitPropList m_initExpr;

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

      MemberMap::iterator mi = m_members.begin();
      while( mi != m_members.end() )
      {
         delete mi->second;
         ++mi;
      }

   }
};



FalconClass::Property::Property( const String& name, size_t value, Expression* expr ):
   m_name(name),
   m_type(t_prop),
   m_expr( expr ),
   m_preExpr(0)
{
   m_value.id = value;
   m_preExpr = new PCode;
   expr->precompile( m_preExpr );
}


FalconClass::Property::Property( const Property& other ):
   m_type( other.m_type ),
   m_value( other.m_value ),
   m_expr( 0 ),
   m_preExpr( 0 )
{
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_preExpr = new PCode;
      m_expr->precompile( m_preExpr );
   }
}

FalconClass::Property::~Property()
{
   delete m_preExpr;
   delete m_expr;
}

FalconClass::Property::Property( Inheritance* value ):
   m_name(value->className()),
   m_type(t_inh),
   m_expr(0),
   m_preExpr(0)
{
   m_value.inh = value;
}


FalconClass::Property::Property( Function* value ):
   m_name( value->name() ),
   m_type(t_func),
   m_expr(0),
   m_preExpr(0)
{
   m_value.func = value;
}


FalconClass::Property::Property( FalconState* value ):
   m_name( "TODO" ),
   m_type(t_state),
   m_expr(0),
   m_preExpr(0)
{
   m_value.state = value;
}


//=====================================================================
// The class
//

FalconClass::FalconClass( const String& name ):
   OverridableClass("Object" , FLC_CLASS_ID_OBJECT ),
   m_fc_name(name),
   m_init(0),
   m_constructor(0),
   m_shouldMark(false),
   m_hasInitExpr( false ),
   m_hasInit( false ),
   m_missingParents( 0 ),
   m_bPureFalcon( true )   
{
   _p = new Private;   
   m_bIsfalconClass = true;
}


FalconClass::~FalconClass()
{
   delete m_constructor;
   delete _p;
}


FalconInstance* FalconClass::createInstance()
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
   members[name] = new  Property( name, _p->m_propDefaults.length() );
   // the add the init value in the value lists.
   _p->m_propDefaults.append( initValue );

   // is this thing deep? -- if it is so, we should mark it
   if( initValue.isGarbaged() )
   {
      m_shouldMark = true;
   }
   
   return true;
}


bool FalconClass::addProperty( const String& name, Expression* initExpr )
{
   Private::MemberMap& members = _p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property -- and record its insertion
   Property* prop = new Property( name, _p->m_propDefaults.length(), initExpr );
   members[name] = prop;

   // expr properties have a default NIL item.
   _p->m_propDefaults.append( Item() );

   // declare that we need this expression to be initialized.   
   _p->m_initExpr.push_back( prop );
   m_hasInitExpr = true;

   if ( m_init == 0 )
   {
      m_init = new SynFunc( "init" );
      m_init->methodOf(this);
      // but this won't modify the m_hasInit state.
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
   members[name] = new Property( name, _p->m_propDefaults.length() );
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
   members[name] = new Property( mth );
   overrideAddMethod( mth->name(), mth );
   return true;
}


Class* FalconClass::getParent( const String& name ) const
{
   Private::MemberMap::const_iterator iter = _p->m_members.find(name);
   if( iter != _p->m_members.end() )
   {
      if( iter->second->m_type == Property::t_inh )
      {
         return iter->second->m_value.inh->parent();
      }
   }

   return 0;
}

  
bool FalconClass::setInit( Function* init )
{
   if( m_init == 0 )
   {
      m_init = init;
      m_init->methodOf(this);
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
   members[name] = new Property( inh );

   // increase the number of unresolved parents?
   if( inh->parent() == 0 )
   {
      m_missingParents ++;
   }
   else if( ! inh->parent()->isFalconClass() )
   {
      m_bPureFalcon = false;
   }
   
   _p->m_inherit.push_back( inh );
   inh->owner( this );

   return true;
}

void FalconClass::onInheritanceResolved( Inheritance* inh )
{
   m_missingParents--;

   if( ! inh->parent()->isFalconClass() )
   {
      m_bPureFalcon = false;
   }
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
   Property& prop = *iter->second;
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
   return iter->second;
}

void FalconClass::gcMarkMyself( uint32 mark ) const
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
   _p->m_members[name] = new Property( state );

   return true;
}


void FalconClass::enumeratePropertiesOnly( PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      if( iter->second->m_type == Property::t_prop)
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


bool FalconClass::isDerivedFrom( Class* cls ) const
{
   // are we the required class?
   if( this == cls ) return true;
   
   Private::ParentList::const_iterator iter = _p->m_inherit.begin();
   while( iter != _p->m_inherit.end() )
   {
      const Inheritance* inh = *iter;
      // ... or is the parent derived from the required class?
      if( inh->parent() != 0 && inh->parent()->isDerivedFrom( cls ) )
      {
         return true;
      }
      ++iter;
   }
   
   return false;
}


void* FalconClass::getParentData( Class* parent, void* data ) const
{
   // The data for all the hierarcy is the same.
   if( this->isDerivedFrom( parent ) )
   {
      return data;
   }
   
   return 0;
}


SynFunc* FalconClass::makeConstructor()
{
   if ( m_constructor == 0 )
   {
      m_constructor = new SynFunc( name() + "#constructor" );
      m_constructor->methodOf( this );
   }
   
   return m_constructor;
}


bool FalconClass::construct( bool bHiddenParents )
{
   if( m_bPureFalcon && m_missingParents == 0 )
   {
      // check if the class has really something to construct.
      bool hasSomething = false;

      TRACE( "Constructing FalconClass %s", name().c_ize() );
      SynFunc* ctr = makeConstructor();
      SynTree& st = ctr->syntree();

      // add the inheritance members from last to first.
      Private::ParentList::const_reverse_iterator riter = _p->m_inherit.rbegin();
      while( riter != _p->m_inherit.rend() )
      {
         Inheritance* current = *riter;
         fassert( current->parent() != 0 && current->parent()->isFalconClass() );
         // we wouldn't be pure Falcon classes if all our parent were not FalconClass
         FalconClass* parent = static_cast<FalconClass*>(current->parent());
         // avoid initializing entities without constructors.
         if( parent->constructor() != 0 )
         {
            st.append( new StmtInit( current ) );
            hasSomething = true;
         }

         ++riter;
      }

      // add the properties first-to-last.
      Private::ParentList::const_iterator iter = _p->m_inherit.begin();
      while( iter != _p->m_inherit.end() )
      {
         Inheritance* current = *iter;
         fassert( current->parent() != 0 && current->parent()->isFalconClass() );
         // we wouldn't be pure Falcon classes if all our parent were not FalconClass
         FalconClass* parent = static_cast<FalconClass*>(current->parent());
         // avoid initializing entities without constructors.
         Private::MemberMap::const_iterator miter = parent->_p->m_members.begin();
         while( miter != parent->_p->m_members.end() )
         {
            // do not override newer properties
            if (_p->m_members.find( miter->first ) == _p->m_members.end() )
            {
               // copy the property
               Property* prop = miter->second->clone();
               // if this was an index property, update the index.
               if( prop->m_type == Property::t_prop )
               {
                  // copy the default property
                  _p->m_propDefaults.append( parent->_p->m_propDefaults[ prop->m_value.id ] );
                  // set the new id
                  prop->m_value.id = _p->m_propDefaults.length()-1;
                  if( prop->expression() != 0 )
                  {
                     _p->m_initExpr.push_back( prop );
                  }
               }
               _p->m_members[miter->first] = prop;
            }
            ++miter;
         }
         ++iter;
      }

      if( m_hasInitExpr )
      {
         hasSomething = true;         
         st.append( new PStepInitExpr( this ) );
      }

      if( m_hasInit )
      {
         hasSomething = true;
         st.append( new PStepInit( this ) );
      }

      //! We may destroy the constructor... or not.
      st.append( new StmtReturn( new ExprSelf ) );
      if( ! hasSomething )
      {
         // great, we don't really need a constructor.
         if( ! bHiddenParents || ctr->paramCount() == 0 )
         {
            // ... unless we have some possible parent that we don't know about.
            delete ctr;
            m_constructor = 0;
         }
      }
      
      return true;
   }

   TRACE( "FalconClass %s cannot be constructed", name().c_ize() );

   // no, we can't create this.
   return false;
}


HyperClass* FalconClass::hyperConstruct()
{
   TRACE( "Creating an hyperclass from %s", name().c_ize() );
   HyperClass* nself = new HyperClass( name(), this );

   // pass all our parents to the hyperclass
   Private::ParentList::iterator pi = _p->m_inherit.begin();
   while( pi != _p->m_inherit.end() )
   {
      nself->addParent( *pi );
      ++pi;
   }

   // Now we're complete and pure falcon...
   _p->m_inherit.clear();
   m_missingParents = 0;
   m_bPureFalcon = true;

   //... so we can have our constructor.
   (void) construct( true );

   //... and tell it to our new self.
   // Notice that this is ok even if the constructor is 0.
   // -- if it is, we just don't need any frame.   
   nself->constructor( m_constructor );

   // and give it to our

   return nself;
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


void FalconClass::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.begin();

   FalconInstance* inst = static_cast<FalconInstance*>(self);

   while( iter != members.end() )
   {
      Property* prop = iter->second;
      if( prop->m_type == Property::t_prop )
      {
         cb( iter->first, inst->m_data[prop->m_value.id] );
         ++iter;
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
      Descriptor( const FalconClass* fc, FalconInstance* inst, String& forming, int d, int l ):
         m_class(fc),
         m_inst(inst),
         m_target( forming ),
         m_depth( d ),
         m_maxlen( l )
      {}

      virtual bool operator()( const String& name, bool )
      {
         Item theItem;
         if( m_class->getProperty( name )->m_type == Property::t_prop )
         {
            m_inst->getMember( name, theItem );
            String temp;
            theItem.describe( temp, m_depth-1, m_maxlen );
            if( m_target != "" )
            {
               m_target += ", ";
            }
            m_target += name + "=" + temp;
         }
         return true;
      }

   private:
      const FalconClass* m_class;
      FalconInstance* m_inst;
      String& m_target;
      int m_depth;
      int m_maxlen;
   };

   String temp;
   Descriptor rator( this, inst, temp, depth, maxlen );

   target = fc_name() +"{" ;
   enumerateProperties( instance, rator );
   target += temp + "}";
}


//=========================================================
// Operators.
//

void FalconClass::op_create( VMContext* ctx, int32 pcount ) const
{
  
   static Collector* coll = Engine::instance()->collector();
   
   // we just need to copy the defaults.
   FalconClass* cls = const_cast<FalconClass*>(this);
   FalconInstance* inst = new FalconInstance(cls);
   inst->data().merge(_p->m_propDefaults);

   // we have to invoke the init method, if any
   if( m_constructor != 0 )
   {
      ctx->call( m_constructor, pcount, FALCON_GC_STORE( coll, this, inst ) );
   }
   else
   {
      // nothing to init; just send the self item in the proper stack return
      ctx->stackResult( pcount+1, FALCON_GC_STORE( coll, this, inst ) );
   }
}


void FalconClass::op_getProperty( VMContext* ctx, void* self, const String& propName ) const
{   
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   
   if( ! overrideGetProperty( ctx, self, propName ) )
   {
      if( ! inst->getMember( propName, ctx->topData() ) )
      {
         Class::op_getProperty( ctx, self, propName );
      }
   }
}


void FalconClass::op_setProperty( VMContext* ctx, void* self, const String& propName ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);

   if( ! overrideSetProperty( ctx, self, propName ) )
   {
      inst->setProperty( propName, ctx->opcodeParam(1) );
      ctx->popData();
   }
}


//==============================================================

FalconClass::PStepInitExpr::PStepInitExpr( FalconClass* o ):
   Statement( Statement::custom_t ),
   m_owner(o)
{
   apply = apply_;
   m_step0 = this;
}

void FalconClass::PStepInitExpr::apply_( const PStep* ps, VMContext* ctx )
{
   // supposedly, if we're here, we have been invited -- iexpr.size() > 0
   const PStepInitExpr* step = static_cast<const PStepInitExpr*>(ps);
   const FalconClass::Private::InitPropList& iprops = step->m_owner->_p->m_initExpr;
   register CodeFrame& ccode = ctx->currentCode();
   register int seqId = ccode.m_seqId;

   TRACE( "In %s class pre-init step %d", step->m_owner->name().c_ize(), seqId );
   
   if( seqId > 0 )
   {
      CallFrame& frame = ctx->currentFrame();
      // the top item is the value of the previous expression -- get it.
      Property* previous = iprops[seqId-1];
      fassert( previous->m_type == FalconClass::Property::t_prop );
      fassert( previous->expression() != 0 );


      FalconInstance* inst = static_cast<FalconInstance*>( frame.m_self.asInst() );
      // if the class and the instance are the same, we can trust the ID
      if( inst->origin() == step->m_owner )
      {
         // at the exit of a pcode, the result is in A
         inst->data()[previous->m_value.id] = ctx->topData();
      }
      else
      {
         // otherwise we're sorry, but we must go by name
         inst->setProperty( previous->m_name, ctx->topData() );
      }
      ctx->popData();
   }

   if( ((size_t)seqId) >= iprops.size() )
   {
      // we're done. 
      ctx->popCode();
      return;
   }

   Property* current = iprops[seqId++];
   ccode.m_seqId = seqId;
   ctx->pushCode( current->pexpr() );
}

//==============================================================

FalconClass::PStepInit::PStepInit( FalconClass* o ):
   Statement( Statement::custom_t ),
   m_owner(o)
{
   apply = apply_;
   m_step0 = this;
}


void FalconClass::PStepInit::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepInit* step = static_cast<const PStepInit*>(ps);
   TRACE( "In %s class init step", step->m_owner->name().c_ize() );

   // we're done.
   ctx->popCode();

   // supposedly, if we're here, we have been invited -- m_init != 0.
   CallFrame& frame = ctx->currentFrame();

   for( uint32 i = 0; i < frame.m_paramCount; ++i )
   {
      ctx->pushData( *ctx->param(i) );
   }
   // Finally, prepare the data stack to accept the functions
   ctx->call( step->m_owner->init(), frame.m_paramCount, frame.m_self );
}

}

/* end of falconclass.cpp */
