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

#include <map>
#include <list>

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
   // here, we own only expressions; other things are held elsewhere.
   delete m_expr;
}

//=====================================================================
// The class
//

FalconClass::FalconClass( const String& name ):
   m_name(name),
   m_shouldMark(false),
   m_init(0)
{
   _p = new Private;
}


FalconClass::FalconClass():
   m_shouldMark(false),
   m_init(0)
{
   _p = new Private;
}


FalconClass::~FalconClass()
{
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


bool FalconClass::addProperty( const String& name, Expression* initExpr )
{
   Private::MemberMap& members = _p->m_members;

   // first time around?
   if ( members.find( name ) != members.end() )
   {
      return false;
   }

   // insert a new property
   _p->m_members.insert( std::make_pair(name, Property( _p->m_propDefaults.length(), initExpr ) ) );
   _p->m_propDefaults.append( Item() );
   
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


bool FalconClass::getMember( const String& name, Item& target ) const
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

      case Property::t_expr:
         // the default value of this entity is nil...
         // ... as the value is defined by the expression at runtime.
         target.setNil();
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


const FalconClass::Property* FalconClass::getMember( const String& name ) const
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


void FalconClass::enumerateMembers( PropertyEnumerator& cb ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.begin();
   while( iter != members.end() )
   {
      if( ! cb( iter->first, ++iter == members.end() ) )
      {
         break;
      }
      else
      {
         ++iter;
      }
   }
}

void FalconClass::enumerateProperties( PropertyEnumerator& cb ) const
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

bool FalconClass::hasMember( const String& name ) const
{
   Private::MemberMap& members = _p->m_members;
   Private::MemberMap::iterator iter = members.find( name );

   return members.end() != iter;
}

}

/* end of falconclass.cpp */
