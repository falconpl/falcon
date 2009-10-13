/*
   FALCON - The Falcon Programming Language.
   FILE: proptable.h

   Very simple double entry table for pure properties.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 14 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
#include <falcon/reflectfunc.h>


namespace Falcon
{

class CoreObject;

/** Descriptor of single property.
* This structure descrbes the contents of a single property;
* It also stores all the data needed for reflection.
*/
struct PropEntry
{
	enum {
		NO_OFFSET = 0xFFFFFFFF
	};

   /** Name of this property */
   const String *m_name;

   /** True if this property is read-only */
   bool m_bReadOnly;

   /** True if this property is write-only. 
	   
	   If this property is accessed via accessors, but has only the set accessor,
	   it cannot be read back (not even at C++ level). Doing so would cause
	   an access error (for write-only read access).

	   \return true if this property is write-only.
   */
   bool isWriteOnly() const { 
	   return m_eReflectMode == e_reflectSetGet && m_reflection.gs.m_getterId == NO_OFFSET; 
   }

   /** Reflection mode. */
   t_reflection m_eReflectMode;

   /** Module-specific property reflection data */
   void *reflect_data;

   /** Default value of the property */
   Item m_value;

   union {
      uint32 offset;
      struct {
         reflectionFunc to;
         reflectionFunc from;
      } rfunc;

      struct {
         uint32 m_getterId;
         uint32 m_setterId;
      } gs;
   } m_reflection;

   /** Reflects a single property
      Item -> user_data
   */
   void reflectTo( CoreObject *instance, void *user_data, const Item &prop ) const;

   /** Reflects a single property.
      user_data -> Item
   */
   void reflectFrom( CoreObject *instance, void *user_data, Item &prop ) const;
};


/** Very simple double entry table for pure properties.

   Property tables are a convenient way to store efficiently and search at the fastest possible
   speed a set of pre-defined pure Falcon strings stored in a safe memory area
   that does not need procetion, collecting or reference counting.

   Unluckily, this is a quite rare situation. Luckily, this is exactly the situation of
   Falcon objects (and classes), whose property names are allocated as Falcon Strings
   in the same Module that declares them.

   The property table stores also informations about the class that declared the property.
   In fact, in case of inheritance, the property may come from an inner class, that needs
   to be managed through a different handler.

   The needed information is the ID of the class in the inheritance list, which is used to
   pick the correct user_data in the object instance, and the user_data handler (ObjectHandler).

   Actually, the ObjectHandler information would not be needed, but it is kept here for caching.
   There is only one property table per class in each program, so it's an affordable cost.

   Once created, the m_value field of each entry is read-only. It stores enumeration and constant
   initialization values for properties and method/class entries for methods/class accessors.

   A property table has two characteristics that are accounted at its creation:
   - It's reflective if it has at least a reflective property.
   - It's static if all of its properties are either reflective or read-only.

*/
class FALCON_DYN_CLASS PropertyTable: public BaseAlloc
{
public:
   uint32 m_size;
   uint32 m_added;
   bool m_bReflective;
   bool m_bStatic;
   PropEntry *m_entries;

public:

   PropertyTable( uint32 size );
   PropertyTable( const PropertyTable & );
   ~PropertyTable();

   uint32 size() const { return m_size; }
   uint32 added() const { return m_added; }

   bool isReflective() const { return m_bReflective; }
   bool isStatic() const { return m_bStatic; }

   /** Analyzes the table and sets its properties. */
   void checkProperties();

   bool findKey( const String &key, uint32 &pos ) const;

   PropEntry &getEntry( uint32 pos ) { return m_entries[pos]; }
   const PropEntry &getEntry( uint32 pos ) const { return m_entries[pos]; }

   const Item *getValue( const String &key ) const
   {
       uint32 pos;
       if( findKey( key, pos ) )
         return getValue( pos );
      return 0;
   }

   Item *getValue( uint32 pos ) { return &m_entries[ pos ].m_value; }
   const Item *getValue( uint32 pos ) const { return &m_entries[ pos ].m_value; }
   const String *getKey( uint32 pos ) const { return m_entries[pos].m_name; }

   PropEntry &append( const String *name );

   bool append( const String *key, const Item &itm, bool bReadOnly = false )
   {
      if ( m_added <= m_size ) {
         PropEntry e = append( key );
         e.m_value = itm;
         e.m_bReadOnly = bReadOnly;
         e.m_eReflectMode = e_reflectNone;
         return true;
      }

      return false;
   }

   bool append( const String *key, const Item &itm, t_reflection mode, uint32 offset, bool bReadOnly = false )
   {
      if ( m_added <= m_size ) {
         PropEntry e = append( key );
         e.m_value = itm;
         e.m_bReadOnly = bReadOnly;
         e.m_eReflectMode = mode;
         e.m_reflection.offset = offset;
         return true;
      }

      return false;
   }

   bool append( const String *key, const Item &itm, reflectionFunc func_from, reflectionFunc func_to = 0 )
   {
      if ( m_added <= m_size ) {
         PropEntry e = append( key );
         e.m_value = itm;
         e.m_bReadOnly = func_to == 0;
         e.m_eReflectMode = e_reflectFunc;
         e.m_reflection.rfunc.from = func_from;
         e.m_reflection.rfunc.to = func_to;
         return true;
      }

      return false;
   }


   PropEntry &appendSafe( const String *key  )
   {
      m_entries[m_added].m_name = key;
      PropEntry *ret = m_entries + m_added;
      m_added++;
      return *ret;
   }

   void appendSafe( const String *key, const Item &itm, bool bReadOnly = false  )
   {
      m_entries[m_added].m_name = key;
      m_entries[m_added].m_value = itm;
      m_entries[m_added].m_bReadOnly = bReadOnly;
      m_entries[m_added].m_eReflectMode = e_reflectNone;
      m_added++;
   }

   void appendSafe( const String *key, const Item &itm, t_reflection mode, uint32 offset, bool bReadOnly = false  )
   {
      m_entries[m_added].m_name = key;
      m_entries[m_added].m_value = itm;
      m_entries[m_added].m_bReadOnly = bReadOnly;
      m_entries[m_added].m_eReflectMode = mode;
      m_entries[m_added].m_reflection.offset = offset;
      m_added++;
   }

   void appendSafe( const String *key, const Item &itm, reflectionFunc func_from, reflectionFunc func_to = 0  )
   {
      m_entries[m_added].m_name = key;
      m_entries[m_added].m_value = itm;
      m_entries[m_added].m_bReadOnly = func_to == 0;
      m_entries[m_added].m_eReflectMode = e_reflectFunc;
      m_entries[m_added].m_reflection.rfunc.from = func_from;
      m_entries[m_added].m_reflection.rfunc.to = func_to;
      m_added++;
   }
};

}

#endif

/* end of proptable.h */
