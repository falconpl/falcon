/*
   FALCON - The Falcon Programming Language.
   FILE: collector_history.cpp

   Falcon Garbage Collector -- data history management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 29 Jun 2011 14:01:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/collector.h>
#include <time.h>

#include <list>
namespace Falcon
{
class Collector::DataStatus::Private
{
public:
   typedef std::list<Collector::HistoryEntry*> EntryList;
   EntryList m_entries;

   Private() {}
   ~Private()
   {
      EntryList::iterator iter = m_entries.begin();
      while( iter != m_entries.end() )
      {
         delete *iter;
         ++iter;
      }
   }
};


Collector::DataStatus::DataStatus( void* data ):
   m_data( data ),
   m_bAlive( true ),
   _p( new Private )
{
}


Collector::DataStatus::~DataStatus()
{
   delete _p;
}


void Collector::DataStatus::addEntry( HistoryEntry* e )
{
   _p->m_entries.push_back( e );
}


String Collector::DataStatus::dump()
{
   String s;
   s.A("0x").H( (int64) m_data, true, 16 ).A( m_bAlive ? " alive" : " dead");

   if( ! _p->m_entries.empty() )
   {
      s += ": ";
   }

   Private::EntryList::iterator iter = _p->m_entries.begin();
   while( iter != _p->m_entries.end() )
   {
      s += (*iter)->dump();
      if( ++iter != _p->m_entries.end() )
      {
         s+= "; ";
      }
   }
   
   return s;
}


void Collector::DataStatus::enumerateEntries( EntryEnumerator& r ) const
{
   Private::EntryList::iterator iter = _p->m_entries.begin();
   while( iter != _p->m_entries.end() )
   {
      Collector::HistoryEntry* he = *iter;
      if( ! r( *he, (++iter != _p->m_entries.end()) ) )
      {
         break;
      }
   }
}

//================================================================
// History entry
//

Collector::HistoryEntry::HistoryEntry( action_t action ):
   m_action( action )
{
   m_ticks = (int64)( clock() / (CLOCKS_PER_SEC / 1000) );
}


Collector::HistoryEntry::~HistoryEntry()
{}

//======================================================
// Create
//

Collector::HECreate::HECreate( const String& file, int line, const String& className ):
   HistoryEntry( ea_create ),
   m_file(file),
   m_line( line ),
   m_class( className )
{}

Collector::HECreate::~HECreate()
{}

String Collector::HECreate::dump() const
{
   return String().A(m_class).A(" created on ").
      N( (int64) this->m_ticks ).A(" at ").A( m_file ).A( ":" ).N(m_line);
}

//======================================================
// Mark
//
Collector::HEMark::HEMark():
   HistoryEntry( ea_mark )
{}

Collector::HEMark::~HEMark()
{}

String Collector::HEMark::dump() const
{
   return String().A(" marked on ").N( (int64) this->m_ticks );
}

//======================================================
// Destroy
//

Collector::HEDestroy::HEDestroy():
   HistoryEntry( ea_destroy )
{}

Collector::HEDestroy::~HEDestroy()
{
}

String Collector::HEDestroy::dump() const
{
   return String().A(" destroyed on ").N( (int64) this->m_ticks );
}

}

/* end of collector_history.cpp */
