/*
   FALCON - The Falcon Programming Language.
   FILE: classuser.h

   A class with some automation to help reflect foreign code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classuser.cpp"

#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>

#include <map>
#include <vector>

namespace Falcon {

class ClassUser::Private
{

public:
   
   typedef std::map<String, Property*> PropMap;
   PropMap m_props;
   
   typedef std::vector<Class*> ParentList;
   ParentList m_parents;
   
   Private(){}
   ~Private() {}
   
};


ClassUser::ClassUser( const String& name ):
   Class(name),
   _p( new Private ),
   m_carriedProps(0)
{
}


ClassUser::~ClassUser()
{
   delete _p;
}
   
void ClassUser::add( Property* prop )
{
   _p->m_props[ prop->name() ] = prop;
   if( prop->isCarried() )
   {
      static_cast<PropertyCarried*>(prop)->m_carrierPos = m_carriedProps;
      m_carriedProps++;
   }
      
}

void ClassUser::addParent( Class* parent )
{
   _p->m_parents.push_back( parent );
}


Class* ClassUser::getParent( const String& name ) const
{
   Private::ParentList::iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      if( (*iter)->name() == name )
      {
         return (*iter);
      }
      ++iter;
   }
   
   return 0;
 }
   

void* ClassUser::getParentData( Class* parent, void* data ) const
{
   Private::ParentList::iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      if( (*iter) == parent )
      {
         return data;
      }
      ++iter;
   }
   
   return 0;
}

   
bool ClassUser::isDerivedFrom( Class* cls ) const
{
   if( cls == this ) 
   {
      return true;
   }
   
   Private::ParentList::iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      if( (*iter) == cls )
      {
         return true;
      }
      ++iter;
   }
   
   return false;
}
   

void ClassUser::gcMarkMyself( uint32 mark )
{
   Class::gcMarkMyself(mark);
   Private::ParentList::iterator iter = _p->m_parents.begin();
   while( iter != _p->m_parents.end() )
   {
      (*iter)->gcMarkMyself( mark );
      ++iter;
   }
}

   

void ClassUser::dispose( void* instance ) const
{
   delete static_cast<UserCarrier*>(instance);
}


void* ClassUser::clone( void* instance ) const
{
   return static_cast<UserCarrier*>(instance)->clone();
}


void ClassUser::gcMark( void* instance, uint32 mark ) const
{
   static_cast<UserCarrier*>(instance)->gcMark(mark);
}
   

bool ClassUser::gcCheck( void* instance, uint32 mark ) const
{
   return static_cast<UserCarrier*>(instance)->gcMark() < mark;
}


void ClassUser::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   Private::PropMap::const_iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      Property* prop = iter->second;
      if( ! cb( prop->name(), ++iter == _p->m_props.end() ) ) 
      {
         break;
      }
   }   
}


void ClassUser::enumeratePV( void* instance, PVEnumerator& cb ) const
{
   Private::PropMap::const_iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      Property* prop = iter->second;
      Item value;
      prop->get( instance, value );
      if( ! (value.isFunction() || value.isMethod()) )
      {
         cb( prop->name(), value );          
      }         
      ++iter;
   }
}


bool ClassUser::hasProperty( void*, const String& prop ) const
{
   return _p->m_props.find( prop ) != _p->m_props.end();
}


void ClassUser::describe( void* instance, String& target, int depth, int maxlen ) const
{
   String temp;
   
   target.reserve(128);
   target.size(0);
   
   target.append("Class " );
   target.append(name());

   if( depth == 0 )
   {
       target += "{...}";
   }
   else
   {
      Private::PropMap::const_iterator iter = _p->m_props.begin();
      bool bFirst = true;
      
      target += '{';
      
      while( iter != _p->m_props.end() )
      {
         Property* prop = iter->second;
         Item value;
         prop->get( instance, value );
         
         if( ! (value.isFunction() || value.isMethod()) )
         {
            if( bFirst )
            {
               bFirst = false;
            }
            else
            {
               target += ','; target += ' ';
            }
         
            value.describe( temp, depth-1, maxlen );
            target.append( prop->name() );
            target.append('=');
            target.append(temp);
            temp.size(0);
         }         
         ++iter;
      }
            
      target += '}';
   }
}


void ClassUser::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector(); 
   
   Item* params = ctx->opcodeParams(pcount);
   void* instance = createInstance( params, pcount );
   ctx->stackResult( pcount + 1, FALCON_GC_STORE( coll, this, instance ) );
}


void ClassUser::op_getProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   Private::PropMap::const_iterator iter = _p->m_props.find( prop );
   if ( iter != _p->m_props.end() )
   {
      iter->second->get( instance, ctx->topData() );
   }
   else
   {
      Class* parent = getParent( prop );
      if( parent != 0 )
      {
         void* data = getParentData( parent, instance );
         if( data != 0 )
         {
            ctx->topData().setUser( parent, data );
            return;
         }
      }
      
      // fallback to class get property
      Class::op_getProperty( ctx, instance, prop );
   }   
}


void ClassUser::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   Private::PropMap::const_iterator iter = _p->m_props.find( prop );
   if ( iter != _p->m_props.end() )
   {
      iter->second->set( instance, ctx->opcodeParam(1) );
      ctx->popData();
   }
   else
   {
      Class::op_setProperty( ctx, instance, prop );
   }
}

}

/* end of classuser.cpp */
