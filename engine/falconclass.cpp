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
#include <falcon/item.h>
#include <falcon/synfunc.h>
#include <falcon/function.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/falconstate.h>
#include <falcon/expression.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/expression.h>
#include <falcon/ov_names.h>
#include <falcon/trace.h>
#include <falcon/stdsteps.h>
#include <falcon/attribute.h>
#include <falcon/stdhandlers.h>
#include <falcon/textwriter.h>
#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/stderrors.h>

#include <falcon/psteps/stmtreturn.h>
#include <falcon/psteps/exprself.h>
#include <falcon/psteps/exprparentship.h>
#include <falcon/psteps/exprinherit.h>

#include "falconinstance_private.h"

#include <map>
#include <list>
#include <vector>

namespace Falcon
{


class FalconClass::Private
{
public:

   typedef std::map<String, Property*> MemberMap;
   MemberMap m_origMembers;
   MemberMap m_curMembers;
   MemberMap* m_members;

   typedef std::list<FalconState*> StateList;
   StateList m_origStates;
   StateList m_curStates;
   StateList* m_states;

   typedef std::vector<Property*> InitPropList;
   InitPropList m_initExpr;

   Private()
   {
      m_members = &m_origMembers;
      m_states = &m_origStates;
   }

   ~Private()
   {
      StateList::iterator si = m_states->begin();
      while( si != m_states->end() )
      {
         delete *si;
         ++si;
      }

      MemberMap::iterator mi = m_members->begin();
      while( mi != m_members->end() )
      {
         delete mi->second;
         ++mi;
      }
   }

   void constructing()
   {
      return;

      m_curMembers = m_origMembers;
      m_members = &m_curMembers;
      m_curStates = m_origStates;
      m_states = &m_curStates;
   }
};


FalconClass::Property::Property( const String& name, const Item& value, Expression* expr ):
   m_name(name),
   m_type(t_prop),
   m_expr( expr )
{
   m_dflt = value;
}


FalconClass::Property::Property( const Property& other ):
   m_name( other.m_name ),
   m_type( other.m_type ),
   m_value( other.m_value ),
   m_expr( 0 )
{
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
   }
}

FalconClass::Property::Property( const Property& other, bool copyInitExpr ):
   m_name( other.m_name ),
   m_type( other.m_type ),
   m_value( other.m_value ),
   m_expr( 0 )
{
   m_dflt = other.m_dflt;
   if( copyInitExpr && other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
   }
}

FalconClass::Property::~Property()
{
   delete m_expr;
}

FalconClass::Property::Property( ExprInherit* value ):
   m_name(value->symbol() == 0 ? "?" : value->symbol()->name()),
   m_type(t_inh),
   m_expr(0)
{
   m_value.inh = value;
}


FalconClass::Property::Property( Function* value, bool bStatic ):
   m_name( value->name() ),
   m_type( bStatic ? t_static_func : t_func),
   m_expr(0)
{
   m_value.func = value;
}


FalconClass::Property::Property( FalconState* value ):
   m_name( "TODO" ),
   m_type(t_state),
   m_expr(0)
{
   m_value.state = value;
}


//=====================================================================
// The class
//

FalconClass::FalconClass( const String& name ):
   OverridableClass(name , FLC_ITEM_USER ),
   m_parentship(0),
   m_constructor(0),
   m_shouldMark(false),
   m_hasInitExpr( false ),
   m_hasInit( false ),
   m_missingParents( 0 ),
   m_bPureFalcon( true ),
   m_bConstructed( false ),
   m_initExprStep( this )
{
   _p = new Private;
   m_bIsFalconClass = true;
   m_category = e_c_falconclass;
}


FalconClass::~FalconClass()
{
   delete m_constructor;
   delete _p;
}


void* FalconClass::createInstance() const
{
   TRACE( "Creating an instance of %s", name().c_ize() );

   if( ! m_bConstructed )
   {
      TRACE( "Class %s is not constructed, failing.", name().c_ize() );
      return 0;
   }

   // we just need to copy the defaults.
   FalconInstance* inst = new FalconInstance(this);
   initInstance( inst );

   // someone else will initialize non-defaultable items.
   return inst;
}

bool FalconClass::registerAttributes( VMContext* ctx )
{
   static PStep* attribStep = &Engine::instance()->stdSteps()->m_fillAttribute;

   Private::MemberMap::iterator iter = _p->m_members->begin();
   Private::MemberMap::iterator end = _p->m_members->end();

   bool bDone = false;

   while( iter != end )
   {
      Property* p = iter->second;
      if (p->m_type == Property::t_func )
      {
         Function* func = p->m_value.func;
         uint32 count = func->attributes().size();

         for( uint32 i = 0; i < count; ++i )
         {
            Attribute* attrib = func->attributes().get( i );
            if( attrib->generator() != 0 ) {
               bDone = true;
               ctx->pushData( Item(Attribute::CLASS_NAME, attrib ) );
               ctx->pushCode( attribStep );
               ctx->pushCode( attrib->generator() );
            }
         }
      }
      ++iter;

   }

   return bDone;
}

bool FalconClass::addProperty( const String& name, const Item& initValue, bool bIsStatic )
{
   TRACE1( "Addong a property \"%s\" to class %s with value.", name.c_ize(), m_name.c_ize() );

   if( m_bConstructed )
   {
      TRACE( "Class %s is ALREADY constructed, failing to add property %s.",
               m_name.c_ize(), name.c_ize() );
      return false;
   }

   Private::MemberMap& members = *_p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required default value
   Property* prop = new Property( name, bIsStatic ? Property::t_static_prop : Property::t_prop );
   prop->m_dflt = initValue;
   members[name] = prop;

   // is this thing deep? -- if it is so, we should mark it
   if( initValue.type() >= FLC_ITEM_METHOD )
   {
      m_shouldMark = true;
   }

   return true;
}


bool FalconClass::addProperty( const String& name, Expression* initExpr )
{
   TRACE1( "Adding a property \"%s\" to class %s with expression.", name.c_ize(), m_name.c_ize() );

   if( m_bConstructed )
   {
      TRACE( "Class %s is ALREADY constructed, failing to add property %s.",
               m_name.c_ize(), name.c_ize() );
      return false;
   }

   Private::MemberMap& members = *_p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() || initExpr->parent() != 0 )
   {
      return false;
   }

   // insert a new property -- and record its insertion
   Property* prop = new Property( name, Property::t_prop );
   prop->expression(initExpr);
   members[name] = prop;

   // declare that we need this expression to be initialized.
   _p->m_initExpr.push_back( prop );
   m_hasInitExpr = true;
   m_constructor = makeConstructor();

   // Let's parent the expression so that it understands the evaluation context.
   // notice that although we have a parent now, we're not owned by it.
   initExpr->setParent(&m_constructor->syntree());

   return true;
}


bool FalconClass::addProperty( const String& name )
{
   TRACE1( "Adding a property \"%s\" to class %s.", name.c_ize(), m_name.c_ize() );

   if( m_bConstructed )
   {
      TRACE( "Class %s is ALREADY constructed, failing to add property %s.",
               m_name.c_ize(), name.c_ize() );
      return false;
   }

   Private::MemberMap& members = *_p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   members[name] = new Property( name, Item() );

   return true;
}


bool FalconClass::addMethod( Function* mth, bool bStatic )
{
   return addMethod( mth->name(), mth, bStatic );
}

bool FalconClass::addMethod( const String& name, Function* mth, bool bStatic )
{
   TRACE1( "Adding method \"%s\" to class %s.", name.c_ize(), m_name.c_ize() );

   if( m_bConstructed )
   {
      TRACE( "Class %s is ALREADY constructed, failing to add method %s.",
               m_name.c_ize(), mth->name().c_ize() );
      return false;
   }

   Private::MemberMap& members = *_p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   mth->methodOf( this );
   mth->module(this->module());
   members[name] = new Property( mth, bStatic );
   overrideAddMethod( mth->name(), mth );
   return true;
}

Class* FalconClass::getParent( const String& name ) const
{
   Private::MemberMap::const_iterator iter = _p->m_members->find(name);
   if( iter != _p->m_members->end() )
   {
      if( iter->second->m_type == Property::t_inh )
      {
         return iter->second->m_value.inh->base();
      }
   }

   return 0;
}


bool FalconClass::addParent( ExprInherit* inh )
{
   // we cannot accept parented expressions.
   if( inh->parent() != 0 || inh->symbol() == 0 )
   {
      return false;
   }

   Private::MemberMap& members = *_p->m_members;

   const String& name = inh->symbol()->name();

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   members[name] = new Property( inh );

   // increase the number of unresolved parents?
   if( inh->base() == 0 )
   {
      m_missingParents ++;
   }
   else if( ! inh->base()->isFalconClass() )
   {
      m_bPureFalcon = false;
   }

   // need a new parentship?
   if( m_parentship == 0 )
   {
      m_parentship = new ExprParentship( declaredAt() );
      // use the constructor as syntactic context.
      SynFunc* ctr = makeConstructor();
      m_parentship->setParent( &ctr->syntree() );
   }

   m_parentship->append( inh );
   return true;
}


 bool FalconClass::setParentship( ExprParentship* inh )
 {
    if( m_parentship != 0 || inh->parent() != 0 )
    {
       return false;
    }

    m_bPureFalcon = true;
    m_missingParents = 0;

    // now see what's the situation
    m_parentship = inh;
    for( int i = 0; i < inh->arity(); i++ )
    {
       ExprInherit* ei = static_cast<ExprInherit*>( inh->nth(i) );
       fassert( ei->trait() == Expression::e_trait_inheritance );

       if( ei->base() == 0 )
       {
          m_missingParents++;
       }
       else if( ! ei->base()->isFalconClass() )
       {
          m_bPureFalcon = false;
       }

       (*_p->m_members)[ei->symbol()->name()] = new Property( ei );
    }

    SynFunc* ctr = makeConstructor();
    inh->setParent( &ctr->syntree() );

    return true;
}


bool FalconClass::getProperty( const String& name, Item& target ) const
{
   Private::MemberMap& members = *_p->m_members;
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
      case Property::t_static_prop:
         target = prop.m_dflt;
         break;

      case Property::t_func:
      case Property::t_static_func:
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
   Private::MemberMap& members = *_p->m_members;
   Private::MemberMap::iterator iter = members.find( name );
   // first time around?
   if ( iter == members.end() )
   {
      return 0;
   }

   // determine what we have found
   return iter->second;
}


bool FalconClass::getProperty( const String& name, void* instance, Item& target ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   return inst->getProperty(name, target);
}

bool FalconClass::setProperty( const String& name, void* instance, const Item& target ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   return inst->setProperty(name, target);
}


void FalconClass::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      Mantra::gcMark(mark);

      Private::MemberMap::iterator iter = _p->m_members->begin();
      Private::MemberMap::iterator end = _p->m_members->end();
      while( iter != end )
      {
         Property* prop = iter->second;
         if( prop->m_type == Property::t_prop )
         {
            prop->m_dflt.gcMark(mark);
         }
         ++iter;
      }

      if( m_parentship != 0 )
      {
         m_parentship->gcMark(mark);
      }

      if( m_constructor != 0 )
      {
         m_constructor->gcMark(mark);
      }
   }
}


bool FalconClass::addState( FalconState* state )
{
   Private::MemberMap& members = *_p->m_members;

   const String& name = state->name();

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property with the required ID
   members[name] = new Property( state );

   return true;
}


void FalconClass::enumeratePropertiesOnly( PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = *_p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      if( iter->second->m_type == Property::t_prop)
      {
         if( ! cb( iter->first ) )
         {
            break;
         }
      }
      ++iter;
   }
}


bool FalconClass::isDerivedFrom( const Class* cls ) const
{
   // are we the required class?
   if( this == cls ) return true;
   if( m_parentship ==  0 ) return false;

   for( int i = 0; i < m_parentship->arity(); ++i )
   {
      ExprInherit* inh = static_cast<ExprInherit*>(m_parentship->nth(i));
      if( inh->base() != 0 )
      {
         if( cls->isDerivedFrom(inh->base()) )
         {
            return true;
         }
      }
      else {
         if( inh->symbol()->name() == cls->name() )
         {
            return true;
         }
      }
   }

   return false;
}


void* FalconClass::getParentData( const Class* parent, void* data ) const
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
      m_constructor->setConstructor();
   }

   return m_constructor;
}


bool FalconClass::construct( VMContext* ctx )
{
   TRACE( "FalconClass::construct -- constructing class \"%s\"", this->name().c_ize() );

   if( m_bConstructed )
   {
      TRACE1( "FalconClass::construct -- already constructed \"%s\"", this->name().c_ize() );
      return true;
   }

   // copy the current status
   _p->constructing();

   // perform property flattening.
   if( m_parentship != 0 )
   {
      // add properties top to bottom.
      int len =  m_parentship->arity();
      for( int i = 0; i < len; i++ )
      {
         ExprInherit* inh = static_cast<ExprInherit*>(m_parentship->nth( i ));
         if( inh->symbol() == 0 )
         {
            throw FALCON_SIGN_XERROR( CodeError, e_internal, .extra("Parentship without symbol") );
         }

         if( inh->base() == 0 )
         {
            // first, try a static resolution
            Item* ibase = 0;
            if ( m_module != 0 ) {
               ibase = m_module->resolve(inh->symbol());
            }

            // if not found, try to resolve in our context (our module might be not live yet)
            if( ibase == 0 && m_module != 0 )
            {
               // will throw on failure
               ibase = ctx->resolveSymbol(inh->symbol(), false);
               fassert( ibase != 0 )
            }

            if( ! ibase->isClass() )
            {
               throw FALCON_SIGN_XERROR( LinkError, e_inv_inherit, .extra(inh->symbol()->name() ) );
            }

            Class* bc = static_cast<Class*>(ibase->asInst());
            inh->base(bc);

            m_missingParents--;
            if( ! bc->isFalconClass() )
            {
               m_bPureFalcon = false;
               continue;
            }
         }

         Class* base = inh->base();
         //... and we checked that we're pure falcon, so it should be a falcon class.
         fassert( base->isFalconClass() );
         FalconClass* fbase = static_cast<FalconClass*>(base);

         Private::MemberMap::const_iterator fbpi = fbase->_p->m_members->begin();
         while( fbpi != fbase->_p->m_members->end() )
         {
            Property* bp = fbpi->second;
            if( bp->m_type != FalconClass::Property::t_inh )
            {
               // do we have a property under the same name?
               if( getProperty(fbpi->first) == 0 )
               {
                  // No? -- then add this
                  TRACE1( "FalconClass::construct -- copying \"%s.%s\" in \"%s\"",
                           fbase->name().c_ize(), bp->m_name.c_ize(),
                           this->name().c_ize() );
                  Property* np = new Property(*bp, false);
                  (*_p->m_members)[np->m_name] = np;
               }
            }

            ++fbpi;
         }
      }
   }

   m_bConstructed = true;

   if( m_missingParents > 0 || ! m_bPureFalcon )
   {
      TRACE1( "FalconClass::construct -- failed to construct \"%s\"", this->name().c_ize() );
      return false;
   }

   return true;
}


HyperClass* FalconClass::hyperConstruct()
{
   TRACE( "Creating an hyperclass from %s", name().c_ize() );
   if( m_missingParents > 0 || (m_bConstructed && m_bPureFalcon) )
   {
      TRACE( "FalconClass %s has not the requisites to become an hyperclass", name().c_ize() );
      return 0;
   }

   HyperClass* nself = new HyperClass( this );
   if( m_parentship != 0 )
   {
      // give the ownership of the parentship to the hyperclass.
      nself->setParentship( this->m_parentship, false );
      this->m_parentship = 0;
   }

   // it's now duty of the caller to construct the hyperclass.
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



//=========================================================
// Class management
//

void FalconClass::gcMarkInstance( void* self, uint32 mark ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   inst->gcMark( mark );
}

bool FalconClass::gcCheckInstance( void* self, uint32 mark ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);
   return inst->currentMark() >= mark;
}


void FalconClass::enumerateProperties( void* , PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = *_p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      const String& name = iter->first;
      if( ! cb( name ) )
      {
         break;
      }

      ++iter;
   }
}


void FalconClass::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Private::MemberMap& members = *_p->m_members;
   Private::MemberMap::iterator iter = members.begin();

   FalconInstance* inst = static_cast<FalconInstance*>(self);

   while( iter != members.end() )
   {
      Property* prop = iter->second;
      if( prop->m_type == Property::t_prop )
      {
         Item* itm = inst->getProperty_internal(&prop->m_name);
         fassert( itm != 0 );
         cb( iter->first, *itm );
         ++iter;
      }
   }
}


bool FalconClass::hasProperty( void*, const String& propName ) const
{
   Private::MemberMap& members = *_p->m_members;
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

      virtual bool operator()( const String& name )
      {
         Item theItem;
         String temp;
         const Property* p = m_class->getProperty( name );
         if( p->m_type == Property::t_prop )
         {
            #ifndef NDEBUG
            bool found = m_inst->getProperty( name, theItem );
            fassert2( found, "Required property not found" );
            #else
            m_inst->getProperty( name, theItem );
            #endif
            if( theItem.isMethod() )
            {
               theItem = theItem.asMethodFunction();
            }
            theItem.describe( temp, m_depth-1, m_maxlen );
         }
         else if( p->m_type == Property::t_static_prop ) {
            p->m_dflt.describe( temp, m_depth-1, m_maxlen );
         }

         if( ! temp.empty() )
         {
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

   target = name() +"{" ;
   if( depth == 0 )
   {
      target += "...";
   }
   else {
      enumerateProperties( instance, rator );
   }
   target += temp + "}";
}


void FalconClass::pushInitExprStep( VMContext* ctx )
{
   if( m_hasInitExpr )
   {
      ctx->pushCode( &m_initExprStep );
   }
}


const Class* FalconClass::handler() const
{
   static const Class* cls = Engine::handlers()->metaFalconClass();
   return cls;
}

void FalconClass::storeSelf( DataWriter* wr ) const
{
   // name has been already stored by the metaclass.
   wr->write(m_shouldMark);
   wr->write(m_hasInitExpr);
   wr->write(m_hasInit);
   wr->write(m_bPureFalcon);

   // now write name and type of each member -- for the values, use flatten.
   // first, always save the original members.
   Private::MemberMap* members = &_p->m_origMembers;
   wr->write( (uint32) members->size() );
   Private::MemberMap::iterator pos = members->begin();
   while( pos != members->end() )
   {
      Property* prop = pos->second;
      wr->write( prop->m_name);
      wr->write( (char) prop->m_type );
      ++pos;
   }

   // save the attributes
   attributes().store(wr);
}


void FalconClass::restoreSelf( DataReader* rd )
{
   // name has been already stored by the metaclass.
   rd->read(m_shouldMark);
   rd->read(m_hasInitExpr);
   rd->read(m_hasInit);
   rd->read(m_bPureFalcon);

   // Read the original members.
   uint32 count;
   rd->read( count );
   for( uint32 i = 0; i < count; ++ i )
   {
      String name;
      char type;
      rd->read( name );
      rd->read( type );

      Property* prop = new Property(name, (Property::Type) type );
      _p->m_origMembers[name] = prop;
   }

   m_bConstructed = false;
   // restore the attributes
   attributes().restore(rd);
}


void FalconClass::flattenSelf( ItemArray& flatArray ) const
{
   Private::MemberMap* members = &_p->m_origMembers;
   Private::MemberMap::iterator pos = members->begin();

   flatArray.reserve( members->size() + 5 );

   if( m_constructor != 0 )
   {
      flatArray.append( Item( m_constructor->handler(), m_constructor ) );
   }
   else {
      flatArray.append( Item() );
   }

   if( m_parentship != 0 )
   {
      flatArray.append( Item( m_parentship->handler(), m_parentship ) );
   }
   else {
      flatArray.append( Item() );
   }

   while( pos != members->end() )
   {
      Property* prop = pos->second;
      switch( prop->m_type )
      {
         case Property::t_prop:
         case Property::t_static_prop:
            flatArray.append( prop->m_dflt );
            if( prop->expression() != 0 )
            {
               flatArray.append( Item( prop->expression()->handler(), prop->expression()));
            }
            else {
               flatArray.append( Item() );
            }
            break;

         case Property::t_func:
         case Property::t_static_func:
            flatArray.append( Item( prop->m_value.func->handler(), prop->m_value.func ) );
            break;

         case Property::t_inh:
            flatArray.append( Item( prop->m_value.inh->handler(), prop->m_value.inh ) );
            break;

         case Property::t_state:
            // TODO
            break;
      }

      ++pos;
   }

   attributes().flatten(flatArray);
}


void FalconClass::unflattenSelf( ItemArray& flatArray )
{
   Private::MemberMap* members = ! m_bConstructed ? &_p->m_origMembers : _p->m_members;
   TRACE( "FalconClass::unflattenSelf -- %s (%d props) %s",
            this->name().c_ize(), flatArray.length(),
            (m_bConstructed ? "constructed" : "not-constructed"));

   Private::MemberMap::iterator pos = members->begin();

   fassert( flatArray.length() >= 2 );

   if( ! flatArray[0].isNil() )
   {
      fassert( flatArray[0].isFunction() );
      m_constructor = static_cast<SynFunc*>(flatArray[0].asInst());
      // constructor status is not flattened.
      m_constructor->setConstructor();
      m_constructor->module(this->module());
   }

   if( ! flatArray[1].isNil() )
   {
      ExprParentship* parents = static_cast<ExprParentship*>(flatArray[1].asInst());
      setParentship(parents);
   }

   _p->m_initExpr.clear(); // just in case
   uint32 count = 2;
   while( pos != members->end() )
   {
      fassert( count < flatArray.length() );
      Property* prop = pos->second;
      switch( prop->m_type )
      {
         case Property::t_prop:
         case Property::t_static_prop:
            prop->m_dflt = flatArray[count++];
            if( flatArray[count].isUser() ) {
               prop->expression( static_cast<Expression*>( flatArray[count].asInst() ));
               _p->m_initExpr.push_back( prop );
               m_hasInitExpr = true;
            }
            break;

         case Property::t_func:
         case Property::t_static_func:
            prop->m_value.func = static_cast<Function*>(flatArray[count].asInst());
            overrideAddMethod( prop->m_name, prop->m_value.func );
            prop->m_value.func->methodOf(this);
            prop->m_value.func->module(this->module());
            break;

         case Property::t_inh:
            prop->m_value.inh = static_cast<ExprInherit*>(flatArray[count].asInst());
            break;

         case Property::t_state:
            // TODO
            break;
      }

      ++pos;
      ++count;
   }

   attributes().unflatten(flatArray, count);
}


void FalconClass::internal_callprop( VMContext* ctx, void* instance, FalconClass::Property& prop, int32 pCount ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);

   if( prop.m_type == FalconClass::Property::t_func )
   {
      ctx->callInternal(prop.m_value.func, pCount, ctx->opcodeParam(pCount));
   }
   else if( prop.m_type == FalconClass::Property::t_prop ) {
      if( pCount > 0 ) {
         ctx->popData( pCount-1 );
         Item temp = ctx->topData();
         inst->_p->m_data[ &prop.m_name ].copyFromLocal( temp );
         ctx->popData();
         ctx->topData() = temp;
      }
      else {
         ctx->addDataSlot();
         ctx->topData().copyFromRemote(inst->_p->m_data[ &prop.m_name ]);
      }
   }
}

void FalconClass::op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   Item delegated;

   if( message != "delegate" )
   {
      DelegateMap* dlg = &inst->_p->m_delegates;

      if( dlg != 0 && dlg->getDelegate(message, delegated) )
      {
         ctx->opcodeParam(pCount) = delegated;
         Class* cls;
         void* inst;
         delegated.forceClassInst(cls, inst);
         cls->op_summon(ctx, inst, message, pCount, bOptional);
         return;
      }
   }

   Private::MemberMap::iterator iter = _p->m_members->find( message );
   if( iter != _p->m_members->end() )
   {
      Property& prop = *iter->second;
      internal_callprop( ctx, instance, prop, pCount );
      return;
   }

   Class::op_summon(ctx, instance, message, pCount, bOptional);
}


void FalconClass::delegate( void* instance, Item* target, const String& message ) const
{
   FalconInstance* mantra = static_cast<FalconInstance*>(instance);

   if( target == 0 )
   {
      mantra->_p->m_delegates.clear();
   }
   else if( target->isUser() && target->asInst() == instance )
   {
      mantra->_p->m_delegates.clearDelegate(message);
   }
   else {
      mantra->_p->m_delegates.setDelegate(message, *target);
   }
}


//=========================================================
// Operators.
//
bool FalconClass::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   static StdSteps& st = *Engine::instance()->stdSteps();

   // we just need to copy the defaults.
   FalconInstance* inst = static_cast<FalconInstance*>(instance);

   // we have to invoke the init method, if any
   if( m_constructor != 0 )
   {
      ctx->pushCode( &st.m_pop );
      Item self( this, inst );
      ctx->insertData( pcount, &self, 1, 0 );
      ctx->callInternal( m_constructor, pcount, Item( this, inst ) );

      // now that we are in the constructor context, we can push the property initializer
      // if we're a base class, we don't need to do this, because property initializers
      // -- are flattened in the topmost child class.
      if( m_hasInitExpr )
      {
         ctx->pushCode( &m_initExprStep );
      }

      // now that we are in the constructor context, we can invoke the inherit sequence.
      if( m_parentship != 0 )
      {
         ctx->stepIn( m_parentship );
      }

      // the constructor goes deep, and will pop the parameters.
      return true;
   }

   return false;
}


void FalconClass::op_getProperty( VMContext* ctx, void* self, const String& propName ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);

   //if( ! overrideGetProperty( ctx, self, propName ) )
   {
      const Item* value = 0;
      if( inst->origin() == this )
      {
         value = inst->_p->getProperty( &propName );
      }
      else {
         const Property* p = getProperty(propName);
         if( p != 0 && p->m_type == Property::t_func )
         {
            ctx->topData().methodize(p->m_value.func);
            return;
         }
         else {
            value = inst->_p->getProperty( &propName );
         }
      }

      if( value != 0 )
      {
         Item temp;
         temp.copyFromRemote(*value);
         if( temp.isMethod() )
         {
            ctx->topData().methodize(temp.asMethodFunction());
         }
         else {
            ctx->topData() = temp;
         }
      }
      else {
         Private::MemberMap::const_iterator iter = _p->m_members->find(propName);
         if( iter != _p->m_members->end() )
         {
            Property* prop = iter->second;
            if( prop->m_type == Property::t_static_prop )
            {
               ctx->topData().copyFromRemote(prop->m_dflt);
               return;
            }
            else if( prop->m_type == Property::t_static_func ) {
               // use the instance as self, as we have a static function accessed via instance.
               // when accesses via class, the class will be the self.
               ctx->topData().setUser(const_cast<FalconClass*>(this), inst);
               ctx->topData().methodize(prop->m_value.func);
               return;
            }
         }
         // Default to base class
         Class::op_getProperty( ctx, self, propName );
      }
   }
}


void FalconClass::op_setProperty( VMContext* ctx, void* self, const String& propName ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(self);

   //if( ! overrideSetProperty( ctx, self, propName ) )
   {
      Item* value = inst->_p->getProperty( &propName );
      if ( value != 0 )
      {
         Item& top = ctx->opcodeParam(1);
         if( top.isFunction() )
         {
            Function* func = top.asFunction();
            ctx->topData().methodize(func);
            value->copyFromLocal(ctx->topData());
         }
         else {
            value->copyFromLocal(top);
            //*value = top;
         }

         ctx->popData();
      }
      else {
         Private::MemberMap::const_iterator iter = _p->m_members->find(propName);
         if( iter != _p->m_members->end() )
         {
            ctx->popData();
            Property* prop = iter->second;
            if( prop->m_type == Property::t_static_prop )
            {
               prop->m_dflt.copyFromLocal(ctx->topData());
               return;
            }
            // we don't allow to override static methods.
         }
         // Default to base class
         Class::op_setProperty( ctx, self, propName );
      }
   }
}


void FalconClass::op_getClassProperty( VMContext* ctx, const String& propName) const
{
   Private::MemberMap::const_iterator iter = _p->m_members->find(propName);
   if( iter != _p->m_members->end() )
   {
      Property* prop = iter->second;
      if( prop->m_type == Property::t_static_prop )
      {
         ctx->topData().copyFromRemote(prop->m_dflt);
      }
      else if( prop->m_type == Property::t_static_func ) {
         // use this class as self.
         ctx->topData().setUser( this->handler(), const_cast<FalconClass*>(this));
         ctx->topData().methodize(prop->m_value.func);
      }
      else {
         throw FALCON_SIGN_XERROR(AccessError, e_static_access, .extra(propName) );
      }
   }
   else{
      // Default to base class
      Class::op_getClassProperty( ctx, propName );
   }
}


void FalconClass::op_setClassProperty( VMContext* ctx, const String& propName ) const
{
   Private::MemberMap::const_iterator iter = _p->m_members->find(propName);
   if( iter != _p->m_members->end() )
   {
      Property* prop = iter->second;
      // we don't allow to override static methods
      // -- so, just check for static properties.
      if( prop->m_type == Property::t_static_prop )
      {
         ctx->popData();
         prop->m_dflt.copyFromLocal(ctx->topData());
      }
      else {
         throw FALCON_SIGN_XERROR(AccessError, e_static_access, .extra(propName) );
      }
   }
   else{
      // Default to base class
      Class::op_setClassProperty( ctx, propName );
   }
}


void FalconClass::render( TextWriter* tw, int32 depth )  const
{
   tw->write( PStep::renderPrefix(depth) );

   // render heading
   tw->write( "class " );
   if( name() != "" && ! name().startsWith("_anon#") )
   {
      tw->write( name() );
   }

   // render the parameters
   Function* ctor = constructor();
   if( ctor != 0 )
   {
      tw->write( "(" );

      const SymbolMap& params = ctor->parameters();
      for( uint32 pc = 0; pc < params.size(); ++pc )
      {
         const Symbol* sym = params.getById(pc);
         if( pc > 0 )
         {
            tw->write( "," );
         }

         tw->write( sym->name() );
      }

      tw->write( ")" );
   }

   // render the parentship (from etc)
   if( m_parentship != 0 )
   {
      m_parentship->render( tw, PStep::relativeDepth(depth) );
   }
   tw->write( "\n" );

   // render the attributes.
   int32 dp = depth < 0 ? -depth : depth+1;

   attributes().render( tw, dp );

   // render the properties
   Private::InitPropList::const_iterator pl_iter = _p->m_initExpr.begin();
   while( pl_iter != _p->m_initExpr.end() )
   {
      Property* prop = *pl_iter;
      tw->write( PStep::renderPrefix(dp) );
      if( prop->isStatic() )
      {
         tw->write( "static " );
      }
      tw->write( prop->m_name );
      tw->write( " = " );
      prop->expression()->render( tw, PStep::relativeDepth(dp) );
      tw->write( "\n" );

      ++pl_iter;
   }

   tw->write( "\n" );

   // and now, the init.
   if( ctor != 0 )
   {
      tw->write( PStep::renderPrefix(dp) );
      tw->write( "init\n");
      ctor->renderFunctionBody(tw, dp + 1 );
      tw->write( PStep::renderPrefix(dp) );
      tw->write( "end\n\n" );
   }

   // then, each method.
   Private::MemberMap::iterator mi_iter = _p->m_members->begin();
   while( mi_iter != _p->m_members->end() )
   {
      const Property* prop = mi_iter->second;
      if ( prop->m_type == Property::t_func || prop->m_type == Property::t_func )
      {
         if( prop->m_type == Property::t_static_func )
         {
            tw->write( "static ");
         }

         prop->m_value.func->render(tw, dp );
         // add an extra \n for elegance.
         tw->write( "\n" );
      }
      ++mi_iter;
   }

   tw->write( PStep::renderPrefix(depth) );
   tw->write( "end" );

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }

}

//==============================================================

FalconClass::PStepInitExpr::PStepInitExpr( FalconClass* o ):
   m_owner(o)
{
   apply = apply_;
}

void FalconClass::PStepInitExpr::describeTo( String& tgt ) const
{
   tgt = "PStepInitExpr for " + m_owner->name();
}

void FalconClass::PStepInitExpr::apply_( const PStep* ps, VMContext* ctx )
{
   // supposedly, if we're here, we have been invited -- iexpr.size() > 0
   const PStepInitExpr* step = static_cast<const PStepInitExpr*>(ps);
   CallFrame& frame = ctx->currentFrame();
   FalconInstance* inst = static_cast<FalconInstance*>( frame.m_self.asInst() );

   applyInitExpr( ctx, step->m_owner, inst );
}


void FalconClass::applyInitExpr( VMContext* ctx, FalconClass* cls, FalconInstance* inst )
{
   const FalconClass::Private::InitPropList& iprops = cls->_p->m_initExpr;
   int size = (int) iprops.size();
   CodeFrame& ccode = ctx->currentCode();
   int& seqId = ccode.m_seqId;

   TRACE1( "In class \"%s\" pre-init step %d/%d", cls->name().c_ize(), seqId, size );

   // Fix in case we're back from a previous run
   if( seqId > 0 )
   {
      Property* prop = iprops[seqId-1];
      TRACE2( "Initializing '%s' from a previous run...", prop->m_name.c_ize() );

      fassert( prop != 0 );
      fassert( prop->m_type == FalconClass::Property::t_prop );
      fassert( prop->expression() != 0 );

      // during init, instances are not shared,
      // we can safely copy them directly.
      inst->_p->m_data[&prop->m_name] = ctx->topData();
      ctx->popData();
   }

   while( seqId < size )
   {

      const Property* prop = iprops[seqId];
      TRACE2( "Initializing property %s at step step %d/%d",
                                       prop->m_name.c_ize(), seqId, size );
      fassert( prop->m_type == FalconClass::Property::t_prop );
      fassert( prop->expression() != 0 );

      // prepare for descent
      seqId++;
      if( ctx->stepInYield( prop->expression(), ccode ) )
      {
         TRACE2( "Descending at step step %d/%d", seqId, size );
         return;
      }

      inst->_p->m_data[&prop->m_name] = ctx->topData();
      ctx->popData();
   }

   // we're done
   ctx->popCode();
}


void FalconClass::initInstance( FalconInstance* inst ) const
{
   Private::MemberMap::const_iterator iter = _p->m_members->begin();
   Private::MemberMap::const_iterator end = _p->m_members->end();

   while( iter != end )
   {
      const Property* prop = iter->second;
      // create the property value

      switch( prop->m_type )
      {
         case FalconClass::Property::t_prop:
            {
               Item& tgt = inst->_p->m_data[&prop->m_name];
               tgt = prop->m_dflt;
            }
            break;

         case FalconClass::Property::t_func:
            {
               Item& tgt = inst->_p->m_data[&prop->m_name];
               tgt.setUser( this, inst );
               tgt.methodize( prop->m_value.func );
            }
            break;

         case FalconClass::Property::t_inh:
            {
               Item& tgt = inst->_p->m_data[&prop->m_name];
               tgt.setUser( prop->m_value.inh->base(), inst );
            }
            break;

         case FalconClass::Property::t_state:
            //TODO
            break;

         // no initialization for statics
         case FalconClass::Property::t_static_prop:
         case FalconClass::Property::t_static_func:
            // Test
            break;
      }

      iter++;
   }
}


void FalconClass::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   inst->makeStorageData(subItems);
}


void FalconClass::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   FalconInstance* inst = static_cast<FalconInstance*>(instance);
   inst->restoreFromStorageData(subItems);
}

}

/* end of falconclass.cpp */
