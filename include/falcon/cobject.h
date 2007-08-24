/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cobject.h
   $Id: cobject.h,v 1.8 2007/08/11 00:11:51 jonnymind Exp $

   Core Object file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004
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
   Core Object file
*/

#ifndef flc_cobject_H
#define flc_cobject_H

#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/common.h>
#include <falcon/proptable.h>
#include <falcon/userdata.h>

namespace Falcon {

class VMachine;

class FALCON_DYN_CLASS CoreObject: public Garbageable
{
   PropertyTable m_properties;
   uint64 m_attributes;
   Symbol *m_instanceOf;
   UserData *m_user_data;

public:

   /** Accepts a pre-existing set of properties as base for this constructor.
      The properties must NOT be already garbage-collector managed, as they are one
      with the core object and will be eventually deleted at object descruction.
   */
   CoreObject( VMachine *vm, const PropertyTable &original, uint64 attribs, Symbol *inst );

   /** Creates a dummy core object.
      This is a constructor that creates a classless and propertyless core object.
      It is useful to stuff some user data in a variable or in the stack, and let GC
      to mark and dispose it as if it were an object.

      I know we must find a better way (i.e. a garbageable dummy), but at the time
      I did this it was the best compromise in term of efficiency, spanwidth of code touched
      and urgent needs.
   */
   CoreObject( VMachine *mp, UserData *ud );

   ~CoreObject()
   {
      delete m_user_data;
   }


   Symbol *instanceOf() const  { return m_instanceOf; }
   bool derivedFrom( const String &className ) const;

   uint64 attributes() const { return m_attributes; }
   void attributes( uint64 data ) { m_attributes = data; }
   void addAttributes( uint64 data ) { m_attributes |= data; }
   bool testAttribute( uint64 value ) const { return (m_attributes & value ) == value; }

   /** Size of the object.
      This is the count of properties in the object. Is it useful? Don't know...
   */
   uint32 size() { return static_cast<uint32>( m_properties.size() ); }

   bool setProperty( const String &prop, const Item &value );
   bool setPropertyRef( const String &prop, Item &value );
   bool setProperty( const String &prop, const String &value );
   bool setProperty( const String &prop, int64 value );
   bool setProperty( const String &prop, int32 value )
   {
      return setProperty( prop, (int64) value );
   }


   /** Returns the a shallow item copy of required property.
      The copy is shallow; strings, arrays and complex data inside
      the original property are shared.

      \param key the property to be found
      \param ret an item containing the object proerty copy.
      \return true if the property can be found, false otherwise
   */
   bool getProperty( const String &key, Item &ret ) const;

   /** Returns the pointer to the physical item position.
      Use with care.

      \param key the property to be found
      \return the property poitner if the object provides the property, 0 otherwise.
   */
   Item *getProperty( const String &key );

   /** Returns a method from an object.
       This function searches for the required property; if it's found,
       and if it's a callable function, the object fills the returned
       item with a "method" that may be directly called by the VM.

       \note Actually this function calls methodize() on the retreived item.
       The value of the \b method parameter may change even if this call
       is unsuccesfull.

       \param key the name of the potential method
       \param method an item where the method will be stored
       \return true if the property exists and is a callable item.
   */
   bool getMethod( const String &key, Item &method ) const;

   bool hasProperty( const String &key ) const
   {
      register uint32 pos;
      return m_properties.findKey( &key, pos );
   }

   Item &getPropertyAt( uint32 pos ) const;

   const String &getPropertyName( uint32 pos ) const {
      return *m_properties.getKey( pos );
   }

   uint32 propCount() const { return  m_properties.added(); }

   /** Returns the user data if set.
      \see setUserData()
      \return The user data, if set, or zero.
   */
   UserData *getUserData() const { return m_user_data; }

   /** Set the user data for this object.
      Extension libraries wishing to provide their own opaque structures
      to scripts via objects can use this feature.
      They must derive their structure from "destroyable", which simply
      provides a virtual destructor (thus, sparing the cost of an extra
      pointer per object to store the user-defined data de-allocator).

      Methods for this object can access directly the user data; the
      methods are granted that the VM will feed them with a correct
      object, so in this way the overhead of chekcing the vailidity
      of the data is avoided.

      The data gets destroyed via the standard destroy operator
      when the host object is garbage-collected.

      \param data the user defined data.
   */
   void setUserData( UserData *data ) { m_user_data = data; }

   /** Creates a shallow copy of this item.
      Will return zero if this item as a user-defined data, that is,
      it's not fully disposeable by the language.
      In future, the user data may be garbageable or may offer a clone
      function, so thintgs may be different, but for now the suggestion
      is that to raise an error in case an uncloneable object is cloned.
      \return a shallow copy of this item.
   */
   CoreObject *clone() const ;
};

}

#endif

/* end of flc_cobject.h */
