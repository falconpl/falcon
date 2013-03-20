/*
   FALCON - The Falcon Programming Language.
   FILE: attributemap.cpp

   Structure holding attributes for function, classes and modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 13:14:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/attributemap.cpp"

#include <falcon/attribute.h>
#include <falcon/attributemap.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/treestep.h>

#include <falcon/pstep.h>
#include <falcon/textwriter.h>

#include <map>
#include <vector>

namespace Falcon {

class AttributeMap::Private
{
public:

   class StrPtrCmp {
   public:
      bool operator()( const String* s1, const String* s2 ) const {
         return *s1 < *s2;
      }
   };

   typedef std::map<const String*, Attribute*, StrPtrCmp> AttribMap;
   typedef std::vector<Attribute*> AttribList;

   AttribMap m_map;
   AttribList m_list;

   Private(){}
   ~Private() {
      AttribList::iterator iter = m_list.begin();
      while( iter != m_list.end() ) {
         delete *iter;
         ++iter;
      }
   }

   void addAttribute( Attribute* a ) {
      m_map[&a->name()] = a;
      a->id( m_list.size() );
      m_list.push_back(a);
   }
};


AttributeMap::AttributeMap()
{
   _p = new Private;
}


AttributeMap::~AttributeMap()
{
   delete _p;
}


AttributeMap::AttributeMap( const AttributeMap& other )
{
   _p = new Private;
   const Private::AttribList& l = other._p->m_list;
   uint32 size = l.size();

   for( uint32 i = 0; i < size; ++i )
   {
      const Attribute* attrib = l[i];
      Attribute* mine = new Attribute(*attrib);
      _p->addAttribute(mine);
   }
}


Attribute* AttributeMap::add( const String& name )
{
   Private::AttribMap::iterator iter = _p->m_map.find(&name);
   if( iter != _p->m_map.end() ) {
      return 0;
   }

   Attribute* attrib = new Attribute();
   attrib->name(name);
   _p->addAttribute(attrib);
   return attrib;
}

bool AttributeMap::remove( const String& name )
{
   Private::AttribMap::iterator iter = _p->m_map.find(&name);
   if( iter == _p->m_map.end() ) {
      return false;
   }

   Attribute* attrib = iter->second;
   uint32 id = attrib->id();
   _p->m_map.erase(iter);

   // re-id all the attributes.
   _p->m_list.erase(_p->m_list.begin() + id );
   for( uint32 i = id; i < _p->m_list.size(); ++ i )
   {
      _p->m_list[i]->id(i);
   }

   // kill the attribute
   delete attrib;

   return true;
}


Attribute* AttributeMap::find( const String& name ) const
{
   Private::AttribMap::iterator iter = _p->m_map.find(&name);
   if( iter == _p->m_map.end() ) {
      return 0;
   }
   return iter->second;
}


uint32 AttributeMap::size() const
{
   return _p->m_list.size();
}


Attribute* AttributeMap::get( uint32 id ) const
{
   if( id > _p->m_list.size() ) {
      return 0;
   }

   return _p->m_list[id];
}


void AttributeMap::gcMark(uint32 mark)
{
   Private::AttribList::iterator iter = _p->m_list.begin();
   while( iter != _p->m_list.end() ) {
      Attribute* a = *iter;
      a->gcMark(mark);
      ++iter;
   }
}

void AttributeMap::store( DataWriter* stream ) const
{
   uint32 size = _p->m_list.size();

   stream->write( size );
   for( uint32 i = 0; i < size; ++i )
   {
      Attribute* attrib = _p->m_list[i];
      attrib->store(stream);
   }
}


void AttributeMap::restore( DataReader* stream )
{
   uint32 size;
   stream->read( size );
   for( uint32 i = 0; i < size; ++i )
   {
      Attribute* attrib = new Attribute();
      try {
         attrib->restore(stream);
         _p->addAttribute(attrib);
      }
      catch( ... )
      {
         delete attrib;
         throw;
      }
   }
}


void AttributeMap::flatten( ItemArray& subItems ) const
{
   uint32 count = _p->m_list.size();
   subItems.reserve( subItems.length() + (count * 2) );
   for(uint32 i = 0; i < count; ++ i )
   {
      Attribute* attrib = _p->m_list[i];
      attrib->flatten(subItems);
   }
}


void AttributeMap::unflatten( const ItemArray& subItems, uint32& start )
{
   uint32 count = _p->m_list.size();

   for(uint32 i = 0; i < count; ++ i )
   {
      Attribute* attrib = _p->m_list[i];
      attrib->unflatten(subItems, start);
   }
}


void AttributeMap::render(TextWriter* tw, int32 depth) const
{
   for( uint32 i = 0; i < size(); ++i )
   {
      Attribute* attr = get(i);
      tw->write( PStep::renderPrefix(depth) );
      tw->write(":");
      tw->write(attr->name());
      tw->write( " => ");
      TreeStep* gen = attr->generator();
      if( gen != 0 )
      {
         gen->render( tw, PStep::relativeDepth(depth) );
      }
      else  {
         Class* cls = 0;
         void* data = 0;
         attr->value().forceClassInst(cls, data);
         String temp;
         cls->describe(data, temp, 1, -1);
         tw->write(temp);
      }

      tw->write( "\n" );
   }
}

}

/* end of attribute.cpp */
