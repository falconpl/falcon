/*
   FALCON - The Falcon Programming Language.
   FILE: proptable.h
   $Id: proptable.h,v 1.2 2006/12/05 15:28:47 gian Exp $

   Very simple double entry table for pure properties.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 14 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Very simple double entry table for pure properties.
*/

#ifndef flc_proptable_H
#define flc_proptable_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/basealloc.h>


namespace Falcon
{

/** Very simple double entry table for pure properties.
   Property tables are a convenient way to store efficiently and search at the fastest possible
   speed a set of pre-defined pure C strings stored as ascii values in a safe memory area
   that does not need procetion, collecting or reference counting.

   Unluckily, this is a quite rare situation. Luckily, this is exactly the situation of
   Falcon objects (and classes), whose property table is fixed and allocated as C strings in
   the module string table (where each property is guaranteed to be a zero terminated
   sequence of single-byte chars).
*/
class FALCON_DYN_CLASS PropertyTable: public BaseAlloc
{
public:
   typedef struct t_config {
      uint16 m_offset;
      uint16 m_size;
      bool m_isSigned;
   } config;

   uint32 m_size;
   uint32 m_added;
   const String **m_keys;
   Item *m_values;
   config *m_configs;

public:

   PropertyTable( uint32 size );
   PropertyTable( const PropertyTable & );
   ~PropertyTable();

   uint32 size() const { return m_size; }
   uint32 added() const { return m_added; }

   bool findKey( const String *key, uint32 &pos ) const;
   Item *getValue( uint32 pos ) const { return m_values + pos; }
   const String *getKey( uint32 pos ) const { return m_keys[pos]; }
   bool hasConfig() const { return m_configs != 0; }
   const config &getConfig( uint32 pos ) const { return m_configs[pos]; }

   bool append( const String *key, const Item &itm );
   bool append( const String *key, const Item &itm, const config &cfg );

   void appendSafe( const String *key, const Item &itm )
   {
      m_keys[m_added] = key;
      m_values[m_added] = itm;
      m_added++;
   }

   void appendSafe( const String *key )
   {
      m_keys[m_added] = key;
      m_values[m_added].setNil();
      m_added++;
   }

   void appendSafe( const String *key, const Item &itm, const config &cfg );
};

}

#endif

/* end of proptable.h */
