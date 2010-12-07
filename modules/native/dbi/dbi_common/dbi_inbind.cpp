/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_bind.cpp

   Database Interface
   Helper for general Falcon-to-C variable binding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 23:47:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_inbind.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/timestamp.h>
#include <falcon/memory.h>
#include <falcon/itemarray.h>
#include <falcon/membuf.h>

#include <falcon/dbi_error.h>

#include <stdio.h>

namespace Falcon {

//=========================================================
// Default time converter
//=========================================================

void DBITimeConverter_ISO::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( bufsize > 19 );

   sprintf( (char*) buffer, "%4.0d-%2.0d-%2.0d %2.0d:%2.0d:%2.0d",
         ts->m_year, ts->m_month, ts->m_day,
         ts->m_hour, ts->m_minute, ts->m_second );

   bufsize = 19;
}
DBITimeConverter_ISO DBITimeConverter_ISO_impl;

//=========================================================
// Default string converter
//=========================================================

char* DBIStringConverter_UTF8::convertString( const String& str, char* target, int &bufsize ) const
{
   char *ret;

   int maxlen = str.length() * 4 + 1;
   if( maxlen <= bufsize )
   {
      // Ok, we can use the buffer
      ret = target;
   }
   else
   {
      ret = (char *) memAlloc( maxlen );
   }

   while( (bufsize = str.toCString( ret, maxlen )) < 0 )
   {
      maxlen *= 2;
      if ( ret != target )
         memFree(ret);
      ret = (char *) memAlloc( maxlen );
   }

   return ret;
}

DBIStringConverter_UTF8 DBIStringConverter_UTF8_impl;



char* DBIStringConverter_WCHAR::convertString( const String& str, char* target, int &bufsize ) const
{
   wchar_t *ret;

   int maxlen = str.length() * 2;
   if( maxlen <= bufsize )
   {
      // Ok, we can use the buffer
      ret = (wchar_t*) target;
   }
   else
   {
      ret = (wchar_t *) memAlloc( maxlen );
   }

   while( (bufsize = str.toWideString( ret, maxlen )) < 0 )
   {
      maxlen *= 2;
      if ( ret != (wchar_t*) target )
         memFree(ret);
      ret = (wchar_t *) memAlloc( maxlen );
   }

   return (char*) ret;
}

DBIStringConverter_WCHAR DBIStringConverter_WCHAR_impl;

//=========================================================
// Single item binding converter
//=========================================================

DBIBindItem::DBIBindItem():
      m_type( t_nil ),
      m_buflen(0)
{
}

DBIBindItem::~DBIBindItem()
{
   clear();
}

void DBIBindItem::set(const Item& value, const DBITimeConverter& tc, const DBIStringConverter& sc )
{
   clear();

   switch( value.type() )
   {
   case FLC_ITEM_NIL:
      break;

   case FLC_ITEM_BOOL:
      m_type = t_bool;
      m_cdata.v_bool = value.asBoolean();
      break;

   case FLC_ITEM_INT:
      m_type = t_int;
      m_cdata.v_int64 = value.asInteger();
      break;

   case FLC_ITEM_NUM:
      m_type = t_double;
      m_cdata.v_double = (double) value.asNumeric();
      break;

   case FLC_ITEM_STRING:
      m_type = t_string;
      m_buflen = bufsize;
      m_cdata.v_string = sc.convertString( *value.asString(), m_buffer, m_buflen );
      break;

   case FLC_ITEM_MEMBUF:
      m_type = t_buffer;
      m_buflen = value.asMemBuf()->length();
      m_cdata.v_buffer = value.asMemBuf()->data();
      break;

   case FLC_ITEM_OBJECT:
      {
         CoreObject* obj = value.asObjectSafe();
         if( obj->derivedFrom( "TimeStamp" ) )
         {
            m_type = t_time;
            TimeStamp* ts = static_cast<TimeStamp*>( obj->getFalconData() );
            m_buflen = bufsize;
            tc.convertTime( ts, m_buffer, m_buflen );
            m_cdata.v_buffer = m_buffer;
            break;
         }
      }
      // else, fall through

   default:
      {
         VMachine* vm = VMachine::getCurrent();
         String str;
         if( vm != 0 )
         {
            vm->itemToString( str, &value );
         }
         else
         {
            str = "<unknown>";
         }

         m_type = t_string;
         m_buflen = bufsize;
         m_cdata.v_string = sc.convertString( str, m_buffer, m_buflen );
      }
   }
}


void DBIBindItem::clear()
{
   if ( m_type == t_string )
   {
      if ( m_cdata.v_string != m_buffer )
      {
         memFree( m_cdata.v_string );
      }
      m_buflen = 0;
   }

   m_type = t_nil;
}

//=========================================================
// Bindings for a whole ItemArray
//=========================================================

DBIInBind::DBIInBind( bool bAc ):
      m_ibind(0),
      m_bAlwaysChange( bAc ),
      m_size(0)
{
}


DBIInBind::~DBIInBind()
{
   delete[] m_ibind;
}


void DBIInBind::unbind()
{
   if ( m_size == 0 )
   {
      m_size = -1;
      return;
   }

   if( m_size != -1 )
   {
      // time to explode.
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_BIND_SIZE, __LINE__ )
            .extra(
              String("").N( (int64) m_size ).A(" != ").N( (int64) 0 )
            )
         );
   }
}


void DBIInBind::bind( const ItemArray& arr,
      const DBITimeConverter& tc,
      const DBIStringConverter& sc )
{
   bool bFirst;

   // as a marker of having completed the first loop, we'll change m_size only at the end
   int nSize;

   // first time around?
   if ( m_ibind == 0 )
   {
      bFirst = true;
      nSize = arr.length();
      m_ibind = new DBIBindItem[nSize];
      onFirstBinding( nSize );
   }
   else
   {
      bFirst = false;
      nSize = m_size;

      if( ((uint32)nSize) != arr.length() )
      {
         // time to explode.
         throw new DBIError( ErrorParam( FALCON_DBI_ERROR_BIND_SIZE, __LINE__ )
               .extra(
                 String("").N( (int64) nSize ).A(" != ").N( (int64) arr.length() )
               )
            );
      }
   }

   // force recalculation of the items.
   if(m_bAlwaysChange )
   {
      bFirst = true;
   }

   for( int i = 0; i < nSize; ++i )
   {
      DBIBindItem& bi = m_ibind[i];
      DBIBindItem::datatype type = bi.type();
      void* buffer = bi.databuffer();
      int len = bi.length();

      bi.set( arr[i], tc, sc );

      // first time around, or changed buffer?
      if( bFirst || bi.type() != type || bi.databuffer() != buffer || bi.length() != len )
      {
         // let the engine determine if the type is compatible with the type of column
         onItemChanged( i );
      }
   }

   // Mark the first loop as complete.
   m_size = nSize;
}

}

/* end of dbi_bind.cpp */
