/*
   FALCON - The Falcon Programming Language.
   FILE: flexydict.cpp

   Standard item type for flexy dictionaries of property-items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 16:55:07 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/flexydict.h>

#include <map>
#include <set>

namespace Falcon {

class FlexyDict::Private
{
public:
   typedef std::map<String, Item> ItemMap;
   ItemMap m_im;
};


FlexyDict::FlexyDict():
   _p( new Private ),
   m_currentMark(0),
   m_flags(0),
   m_meta(0),
   m_bOwnMeta(false)
{
}

FlexyDict::FlexyDict( const FlexyDict& other ):
   _p( new Private ),
   m_currentMark(other.m_currentMark),
   m_flags(other.m_flags),
   m_meta(0),
   m_bOwnMeta(false),
   m_base(other.m_base)
{
   _p->m_im = other._p->m_im;
   if (other.m_bOwnMeta && other.m_meta != 0 )
   {
      m_bOwnMeta = true;
      m_meta = new FlexyDict( *other.m_meta );
   }
   else  {
      m_bOwnMeta = false;
      m_meta = other.m_meta;
   }
}

FlexyDict::~FlexyDict()
{
   delete _p;

   if( m_bOwnMeta )
   {
      delete m_meta;
   }
}


void FlexyDict::gcMark( uint32 mark )
{
   if( m_currentMark == mark )
   {
      return;
   }

   Private::ItemMap::iterator pos = _p->m_im.begin();
   while( pos != _p->m_im.end() )
   {
      const Item& value = pos->second;
      value.gcMark( mark );
      ++pos;
   }

   if( m_meta != 0 )
   {
      m_meta->gcMark(mark);
   }

   m_base.gcMark( mark );
}


void FlexyDict::enumerateProps( Class::PropertyEnumerator& e ) const
{
   Private::ItemMap::const_iterator pos = _p->m_im.begin();

   if( m_base.length() == 0 )
   {
      while( pos != _p->m_im.end() )
      {
         const String& key = pos->first;
         e( key, ++pos == _p->m_im.end() );
      }
   }
   else
   {
      // support base class property masking.
      std::set<String> temp;

      while( pos != _p->m_im.end() )
      {
         const String& key = pos->first;
         if( temp.find( key ) == temp.end() )
         {
            e( key, true );
            temp.insert( key );
         }

         ++pos;
      }

      // use an enumerator to support subclasses.
      class Rator: public Class::PropertyEnumerator
      {
      public:
         Rator( std::set<String>& temp, Class::PropertyEnumerator& e ):
            m_temp(temp),
            m_e(e)
         {}
         virtual ~Rator(){}

         virtual bool operator()( const String& property, bool )
         {
            if( m_temp.find( property ) == m_temp.end() )
            {
               bool cont = m_e( property, false );
               m_temp.insert( property );
               return cont;
            }
            return true;
         }
      private:
         std::set<String>& m_temp;
         Class::PropertyEnumerator& m_e;
      };

      Rator rator( temp, e );
      for ( length_t i = 0; i < m_base.length(); ++ i )
      {
         Class* cls;
         void * data;
         m_base[i].forceClassInst( cls, data );
         cls->enumerateProperties( data, rator );
      }
   }
}


void FlexyDict::enumeratePV( Class::PVEnumerator& e )
{
   Private::ItemMap::iterator pos = _p->m_im.begin();
   if( m_base.length() == 0 )
   {
      while( pos != _p->m_im.end() )
      {
         const String& key = pos->first;
         e( key, pos->second );
         ++pos;
      }
   }
   else
   {
      // support base class property masking.
      std::set<String> temp;

      while( pos != _p->m_im.end() )
      {
         const String& key = pos->first;
         if( temp.find( key ) == temp.end() )
         {
            e( key, pos->second );
            temp.insert( key );
         }
         
         ++pos;
      }

      // use an enumerator to support subclasses.
      class Rator: public Class::PVEnumerator
      {
      public:
         Rator( std::set<String>& temp, Class::PVEnumerator& e ):
            m_temp(temp),
            m_e(e)
         {}
         virtual ~Rator(){}

         virtual void operator()( const String& property, Item& value )
         {
            if( m_temp.find( property ) == m_temp.end() )
            {
               m_e( property, value );
               m_temp.insert( property );
            }
         }
      private:
         std::set<String>& m_temp;
         Class::PVEnumerator& m_e;
      };

      Rator rator( temp, e );
      for ( length_t i = 0; i < m_base.length(); ++ i )
      {
         Class* cls;
         void * data;
         m_base[i].forceClassInst( cls, data );
         cls->enumeratePV( data, rator );
      }
   }
}


bool FlexyDict::hasProperty( const String& p ) const
{
   return _p->m_im.find(p) != _p->m_im.end();
}


void FlexyDict::describe( String& target, int depth, int maxlen ) const
{
   if( depth == 0 )
   {
      target = "{...}";
      return;
   }

   String value;
   Private::ItemMap::const_iterator pos = _p->m_im.begin();
   while( pos != _p->m_im.end() )
   {
      const String& key = pos->first;

      if( target.size() > 1 )
      {
         target += ", ";
      }

      value.size(0);
      pos->second.describe( value, depth-1, maxlen );
      target += key + "=" + value;

      ++pos;
   }
}


uint32 FlexyDict::size() const
{
   return _p->m_im.size();
}


Item* FlexyDict::find( const String& value ) const
{
   Private::ItemMap::iterator pos = _p->m_im.find( value );
   if( pos != _p->m_im.end() )
   {
      return &pos->second;
   }
   
   return 0;
}


void FlexyDict::insert( const String& key, Item& value )
{
   _p->m_im[key].assign(value);
}


bool FlexyDict::meta( FlexyDict* fd, bool own )
{
   FlexyDict* metaChild = fd;
   while( metaChild != 0 )
   {
      if ( metaChild == this )
      {
         return false;
      }
      metaChild = metaChild->meta();
   }

   if (m_bOwnMeta )
   {
      delete m_meta;
   }
   m_meta = fd;
   m_bOwnMeta = own;

   return true;
}

}

/* end of flexydict.cpp */
