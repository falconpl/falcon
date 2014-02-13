/*
   FALCON - The Falcon Programming Language.
   FILE: closeddata.cpp

   Data for closures
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 15:39:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/closeddata.cpp"

#include <falcon/trace.h>
#include <falcon/item.h>
#include <falcon/mt.h>
#include <falcon/function.h>
#include <falcon/itemarray.h>
#include <falcon/closeddata.h>
#include <falcon/engine.h>
#include <falcon/refcounter.h>
#include <falcon/stdhandlers.h>
#include <falcon/vmcontext.h>
#include <falcon/symbol.h>

#include <map>
#include <vector>

#include <string.h>

namespace Falcon {
/*

//TODO: Create a separate class for by id and by name association
class Closure::Private
{
public:
   mutable Mutex m_mtx;

   class StringPtrCmp {
   public:
      bool operator() ( const String* p1, const String* p2 ) const {
         return *p1 < *p2;
      }
   };

   class Entry {
   public:
      uint32 id;
      String name;
      Item value;

      Entry( uint32 i, const String& n, const Item& v ):
         id(i),
         name(n),
         value(v)
      {
      }

      Entry( const Entry& other ):
         id( other.id ),
         name( other.name ),
         value( other.value )
      {}
   };

   typedef std::map<const String*, Entry*, StringPtrCmp> EntryMap;
   EntryMap m_data;

   typedef std::vector<Entry*> EntryVector;
   EntryVector m_byid;

   Private() {}

   Private( const Private& other )
   {
      other.m_mtx.lock();
      for(uint32 i = 0; i < other.m_byid.size(); ++i ) {
         Entry* mine = new Entry( *other.m_byid[i] );
         other.m_mtx.unlock();
         m_byid.push_back( mine );
         m_data[&mine->name] = mine;
         other.m_mtx.lock();
      }
      other.m_mtx.unlock();
   }

   ~Private() {
      EntryVector::iterator iter = m_byid.begin();
      EntryVector::iterator end = m_byid.end();
      while( iter != end ) {
         delete *iter;
         ++iter;
      }
   }
};



Closure::Closure():
   m_mark(0)
{
   _p = new Private;
}

Closure::Closure( const Closure& other ):
   m_mark(0)
{
   _p = new Private( *other._p );
}

Closure::~Closure()
{
   delete _p;
}

void Closure::gcMark( uint32 mark )
{
   if( m_mark < mark )
   {
      m_mark = mark;

      _p->m_mtx.lock();
      for(uint32 i = 0; i < _p->m_byid.size(); ++i ) {
         Item& value = _p->m_byid[i]->value;
         _p->m_mtx.unlock();
         value.gcMark(mark);
         _p->m_mtx.lock();
      }
      _p->m_mtx.unlock();
   }
}


uint32 Closure::add( const String& name, const Item& value )
{
   uint32 id = 0;

   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.find(&name);
   if( pos == _p->m_data.end() ) {
      id = _p->m_byid.size();
      Private::Entry* entry = new Private::Entry(id, name, value);
      _p->m_data[&entry->name] = entry;
   }
   else {
      id = pos->second->id;
      pos->second->value = value;
   }

   _p->m_mtx.unlock();

   return id;
}


Item* Closure::get( const String& name ) const
{
   Item* value = 0;
   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.find(&name);
   if( pos != _p->m_data.end() )
   {
      value = &pos->second->value;
   }
   _p->m_mtx.unlock();

   return value;
}


Item* Closure::get( uint32 id ) const
{
   Item* value = 0;
   _p->m_mtx.lock();
   if( id < _p->m_byid.size() )
   {
      value = _p->m_byid[id];
   }
   _p->m_mtx.unlock();

   return value;
}

uint32 Closure::getIdOf( const String& name ) const
{
   uint32* id = 0;
   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.find(&name);
   if( pos != _p->m_data.end() )
   {
      id = pos->second->id;
   }
   _p->m_mtx.unlock();

   return id;
}
*/


class ItemRef: public Item
{
private:

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(ItemRef);
};

class ClosedData::Private
{
public:
   mutable Mutex m_mtx;

   typedef std::map<const Symbol*, ItemRef*> EntryMap;
   EntryMap m_data;

   Private() {}
   Private( const Private& other )
   {
      copy( other );
   }

   ~Private() {
      EntryMap::const_iterator iter = m_data.begin();
      EntryMap::const_iterator end = m_data.end();
      while( iter != end )
      {
         const Symbol* sym = iter->first;
         sym->decref();
         ItemRef* ir = iter->second;
         ir->decref();
         ++iter;
      }
   }

   void copy( const Private& other ) {
      other.m_mtx.lock();
      EntryMap::const_iterator iter = other.m_data.begin();
      EntryMap::const_iterator end = other.m_data.end();
      while( iter != end )
      {
         const Symbol* sym = iter->first;
         ItemRef* oi = iter->second;
         sym->incref();
         oi->incref();
         other.m_mtx.unlock();

         m_mtx.lock();
         EntryMap::iterator pos = m_data.find( iter->first );
         if( pos == m_data.end() ) {
            m_data[sym] = oi;
         }
         else {
            pos->second->copy(*oi);
            sym->decref();
            oi->decref();
         }
         m_mtx.unlock();

         other.m_mtx.lock();
         // items are never deleted, so we're sure the iterator stayed valid
         ++iter;
      }
      other.m_mtx.unlock();
   }
};



ClosedData::ClosedData():
   m_mark(0)
{
   _p = new Private;
}

ClosedData::ClosedData( const ClosedData& other ):
   m_mark(0)
{
   _p = new Private;
   _p->copy( *other._p );
}

ClosedData::~ClosedData()
{
   delete _p;
}


void ClosedData::copy( const ClosedData& other )
{
   _p->copy( *other._p );
}

void ClosedData::gcMark( uint32 mark )
{
   if( m_mark == mark ) {
      return;
   }
   m_mark = mark;

   _p->m_mtx.lock();
   Private::EntryMap::iterator iter = _p->m_data.begin();
   Private::EntryMap::iterator end = _p->m_data.end();
   while( iter != end )
   {
      _p->m_mtx.unlock();
      iter->second->gcMark(mark);
      _p->m_mtx.lock();
      // items are never deleted, so we're sure the iterator stayed valid
      ++iter;
   }
   _p->m_mtx.unlock();
}


uint32 ClosedData::size() const
{
   uint32 s;
   _p->m_mtx.lock();
   s = _p->m_data.size();
   _p->m_mtx.unlock();

   return s;
}


void ClosedData::add( const String& name, const Item& value )
{
   const Symbol* sym = Engine::getSymbol(name);
   add(sym, value);
   sym->decref();
}

void ClosedData::add( const Symbol* sym, const Item& value )
{
   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.find( sym );
   if( pos == _p->m_data.end() ) {
      ItemRef* ir = new ItemRef();
      ir->copy( value );
      sym->incref();
      _p->m_data[sym] = ir;
   }
   else {
      pos->second->copy(value);
   }

   _p->m_mtx.unlock();
}


Item* ClosedData::get( const String& name ) const
{
   const Symbol* sym = Engine::getSymbol(name);
   Item* value = get(sym);
   sym->decref();

   return value;
}


Item* ClosedData::get( const Symbol* sym ) const
{
   Item* value = 0;
   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.find(sym);
   if( pos != _p->m_data.end() )
   {
      value = pos->second;
   }
   _p->m_mtx.unlock();

   return value;
}


void ClosedData::flatten( VMContext*, ItemArray& subItems ) const
{
   _p->m_mtx.lock();
   uint32 size = _p->m_data.size();
   _p->m_mtx.unlock();

   //just prepare to store the items; we don't care if the map changes now.
   subItems.reserve(subItems.length() + size * 2 + 1);

   _p->m_mtx.lock();
   subItems.append((int64) _p->m_data.size() );
   Private::EntryMap::iterator pos = _p->m_data.begin();
   Private::EntryMap::iterator end = _p->m_data.end();
   while( pos != end )
   {
      const Symbol* sym = pos->first;
      const Item& value = *pos->second;
      subItems.append(Item(sym));
      subItems.append(value);

   }
   _p->m_mtx.unlock();
}


void ClosedData::unflatten( VMContext*, ItemArray& subItems, uint32 pos )
{
   if( pos +2 >= subItems.length() ) {
      return;
   }

   Item& sizeItem = subItems[pos++];
   if( !sizeItem.isInteger()  ) {
      return;
   }
   int64 size = sizeItem.asInteger();

   while( size > 0 && pos + 2 < subItems.length() ) {
      Item& nameItem = subItems[pos++];
      Item& valueItem = subItems[pos++];
      if( ! nameItem.isSymbol() ) {
         return;
      }
      const Symbol* sym = nameItem.asSymbol();

      ItemRef* ir = new ItemRef;
      ir->copy(valueItem);
      _p->m_data[sym] = ir;
      sym->incref();
   }
}


void ClosedData::defineSymbols( VMContext* ctx )
{
   _p->m_mtx.lock();
   Private::EntryMap::iterator pos = _p->m_data.begin();
   Private::EntryMap::iterator end = _p->m_data.end();
   while( pos != end )
   {
      const Symbol* sym = pos->first;
      Item* value = pos->second;
      ctx->defineSymbol( sym , value );
      ++pos;
   }
   _p->m_mtx.unlock();
}

Class* ClosedData::handler() const
{
   static Class* cls = Engine::handlers()->closedDataClass();

   return cls;
}

}

/* end of closeddata.cpp */
