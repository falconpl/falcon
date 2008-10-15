/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cobject.h

   Core Object file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
#include <falcon/objectmanager.h>
#include <falcon/cclass.h>

namespace Falcon {

class VMachine;
class AttribHandler;
class Attribute;

class FALCON_DYN_CLASS CoreObject: public Garbageable
{
   AttribHandler *m_attributes;
   const CoreClass *m_generatedBy;
   void *m_user_data;
   Item *m_cache;

   friend class Attribute;
public:

   /** Creates an object deriving it from a certain class.
      The constructor also creates the needed space to store
      the user_data needed by each subclass of the class.

      A user_data matching with the reflective properties and
      with the class ObjectManager can be provided. If not provided,
      and if the class provides an object manager, the object
      manager init method will be called.

      If an already created user_data is provided and if it's possible
      to cache its data, reflectFrom() method will be called to
      grab user data values into the local item cahce.

      \param cls The core class that instances this object.
      \param user_data Pre created user data or 0.
   */
   CoreObject( const CoreClass *cls, void *user_data = 0);

   ~CoreObject();

   const Symbol *instanceOf() const  { return m_generatedBy->symbol(); }
   bool derivedFrom( const String &className ) const;

   /** Return the head of attribute lists.
      This is used internally by Attribute class give/remove, and
      by serialization.
   */

   AttribHandler *attributes() const { return m_attributes; }

   /** Check if this item has a certain attribute.
      To give/remove an attribute from an object, it is necessary to use
      the Attribute instance you want to give to or remove from this object.
      \param attrib the attribute to be searched for.
      \return true if the object has a certain attribute.
   */
   bool has( const Attribute *attrib ) const;

   /** Check if this item has a certain attribute.
      To give/remove an attribute from an object, it is necessary to use
      the Attribute instance you want to give to or remove from this object.
      \param attrib the name of the attribute to be searched for.
      \return true if the object has a certain attribute.
   */
   bool has( const String &attrib ) const;

   /** Sets a property in the object.
      If the property is found, the value in the item is copied, otherwise the
      object is untouched and false is returned.

      In case of reflected objects, it may be impossible to set the property. In that case,
      the owning vm gets an error, and false is returned.

      \param prop The property to be set.
      \param value The item to be set in the property.
      \return ture if the property can be set, false otherwise.
   */
   bool setProperty( const String &prop, const Item &value );

   /** Stores an arbitrary string in a property.
      The string is copied in a garbageable string created by the object itself.
   */
   bool setProperty( const String &prop, const String &value );


   /** Returns the a shallow item copy of required property.
      The copy is shallow; strings, arrays and complex data inside
      the original property are shared.

      \param key the property to be found
      \param ret an item containing the object proerty copy.
      \return true if the property can be found, false otherwise
   */
   bool getProperty( const String &key, Item &ret );


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
   bool getMethod( const String &propName, Item &method )
   {
      if ( getProperty( propName, method ) )
         return method.methodize( this );
      return false;
   }


   bool hasProperty( const String &key ) const
   {
      register uint32 pos;
      return m_generatedBy->properties().findKey( key, pos );
   }


   const String &getPropertyName( uint32 pos ) const {
      return *m_generatedBy->properties().getKey( pos );
   }

   /** Return the class item that generated this class */

   /** Gets the count of the properties of this object.
      This is useful when iterating on an object properties.
      \return count of properties stored in this instance (can be 0).
   */
   uint32 propCount() const { return m_generatedBy->properties().added(); }

   /** Gets a property at a given position in the property table.
      This is useful when iterating on an object properties.
      The property is also reflected, if necessary, from the underlying
      native user_data.

      \note passing a \b pos param out of range will assert in debug and
            crash in runtime.
      \param pos the property ID to be retreived.
      \param ret An item where to store the value of the property.
      \see propCount()
   */
   void getPropertyAt( uint32 pos, Item &ret );

   /** Sets a property at a given position in the property table.
      This is useful when iterating on an object properties.
      The property is also reflected, if necessary, into the underlying
      native user_data.

      \note passing a \b pos param out of range will assert in debug and
            crash in runtime.
      \param pos the property ID to be set.
      \param ret The valued to be stored.
      \see propCount()
   */
   void setPropertyAt( uint32 pos, const Item &ret );

   /** Creates a shallow copy of this item.
      Will return zero if this item has a non-cloneable user-defined data,
      that is, it's not fully manageable by the language.

      Clone operation requests the class ObjectManager to clone the user_data
      stored in this object, if any. In turn, the ObjectManager may ask the
      user_data, properly cast, to clone itself. If one of this operation
      fails or is not possible, then the method returns 0. The VM will eventually
      raise a CloneError to signal that the operation tried to clone a non
      manageable user-data object.

      If this object has not a user_data, then the cloneing will automatically
      succeed.

      \return a shallow copy of this item.
   */
   CoreObject *clone() const;

   /** Reflect an external data into this object.
      This method uses the reflection informations in the generator class property table
      to load the data stored in the user_data parameter into the local item vector cache.

      If the object has not a cache, the method returns immediately doing nothing.
   */
   void reflectFrom( void *user_data );

   /** Reflect this object into external data.
      This method uses the reflection informations in the generator class property table
      to store a C/C++ copy of the data stored in the local cache of this object into
      an external data.

      If the object has not a cache, the method returns immediately doing nothing.
   */
   void reflectTo( void *user_data );

   /** Returns the user data. */
   void *getUserData() const { return m_user_data; }

   /** Sets the user data.
      Using this method is generally a bad idea. Be sure to know what you're doing.
   */
   void setUserData( void *ud ) { m_user_data = ud; }

   /** Shortcut to checking if the class is a reflective sequence. */
   bool isSequence() const;

   /** Shortcut to access object manager (if the class is reflective). */
   ObjectManager *getObjectManager() const { return m_generatedBy->getObjectManager(); }

   /** Performs GC marking of the inner object data */
   void gcMarkData( byte mark );

   /** Get a pointer to a cached property.
      Will return zero if the pointer ID is out of range or if this object doesen't provide a
      cache.
      \param pos the property ID.
      \return A pointer to the item that caches the given property or zero.
   */
   Item *cachedProperty( const String &name ) const
   {
      if( m_cache == 0 )
         return 0;

      register uint32 pos;
      if ( ! m_generatedBy->properties().findKey( name, pos ) )
         return 0;

      return m_cache + pos;
   }

   /** Get a pointer to a cached item.
      Will return zero if the pointer ID is out of range or if this object doesen't provide a
      cache.
      \param pos the property ID.
      \return A pointer to the item that caches the given property or zero.
   */
   Item *cachedPropertyAt( uint32 pos ) const {
      if ( m_cache == 0 || pos > m_generatedBy->properties().added() )
         return 0;
      return m_cache + pos;
   }

   /** Get the class that generated this object.
      This is the phisical core object instance that generated this object.
      The symbol in the returned class is the value returend by instanceOf().
      \return the CoreClass that were used to generate this object.
   */
   const CoreClass *generator() const { return m_generatedBy; }

   /** Small utility to cache string properties.
      If the object has a cache and the data to be stored is a string coming from
      an external source, then this method can be used to efficiently store a copy
      of the external string in the cache.

      The method creates a new GarbageString only if necessary.
   */
   void cacheStringProperty( const String& propName, const String &value );
};

}

#endif

/* end of flc_cobject.h */
