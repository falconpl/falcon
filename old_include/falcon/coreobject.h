/*
   FALCON - The Falcon Programming Language.
   FILE: coreobject.h

   Base class for all the strict OOP available in falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_cobject_H
#define flc_cobject_H

#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/common.h>
#include <falcon/proptable.h>
#include <falcon/deepitem.h>
#include <falcon/item.h>
#include <falcon/objectfactory.h>

namespace Falcon {

class VMachine;
class String;
class CoreClass;
class MemPool;
class Sequence;
class FalconData;
class ItemDict;

/** Base core object class.

   To create your own objects, derive from this class and reimplement:

   \code
   virtual bool hasProperty( const String &key ) const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;
   virtual CoreObject *clone() const;
   \endcode

   Eventually, reimplement also:

   \code
   virtual bool serialize( Stream *stream, bool bLive ) const;
   virtual bool deserialize( Stream *stream, bool bLive );
   \endcode

   Those default methods in this class will save the object on the stream on live
   serialization (i.e. inter-vm serializations), but will just fail on
   normal serializations.

   Then, create a factory function returning an object of your class
   when creating an instance, like the following:

   \code
   // typedef ObjectFactory ...
   CoreObject* MyClassFactory( const CoreClass *cls, void *data, bool bDeserializing );
   \endcode

   Remember that inner program-specific data may be provided later via setUserData() method,
   so the factory function must take into accunt the fact that \b data may not be provided (may be 0).

   The \b bDeserializing parameter will be true when creating an instance after a
   serialized data. As deserialize() is going to be called next, the constructor may take this
   parameter to avoid performing full construction and let deserialize() to setup the object.


   Then, in the module owning this object, the class that shall return an instance of this
   must be configured via ClassDef::factory( ObjectFactory ) method. In example:

   \code
   Symbol *my_class = self->addClass( "MyClass", my_class_init );
   my_class->getClassDef()->factory( &MyClassFactory );
   \endcode

   Three standard subclasses with their respective factories are provided.
   - FalconObject is the standard CoreObject containing falcon Item instances in each property and
      possibly a FalconData entity.
   - ReflectObject is totally opaque, and all its get/set properties are sent to the class reflection
     table. The class reflection table indicates which action must be taken when setting or getting
     a property, and can both store C native data in the underlying user_data, or call functions to
     perform this task.
   - CRObject: Is a reflective object that has a back-up Item for each property. If a prioperty is
     not declared reflective in the class reflection table, it is treated as if in a FalconObject,
     otherwise its value is generated through the reflection mechanism and cached in the property
     item.

   Those three class factories are automatically applied by the VM in case it has not been set.
   If the class has all the properties fully reflected (or reflected on read and read only)
   ReflectObject factory will be used; if one or more properties are not reflected CRObject
   factory will be used; if none is reflected, FalconObject factory is used.

   \note Actually, there are three factories for each one of the basic classes, depending on
         which kind of user data is expected to be associated with the created instance, if any.
         The VM uses the factory functions that suppose that the data stored in the instance
         will be an instance of FalconData class, as this is the most common case and the only
         data stored in objects by the engine.
*/

class FALCON_DYN_CLASS CoreObject: public DeepItem, public Garbageable
{
protected:
   void *m_user_data;
   bool m_bIsFalconData;
   bool m_bIsSequence;

   const CoreClass *m_generatedBy;

   /** State name. */
   String* m_state;

   CoreObject( const CoreClass *parent );
   CoreObject( const CoreObject &other );

   /** Creates the default value for this property.

      Simple reflection can be done by overloading CoreObject and handling
      reflected properties in a string if/else cascade or in a switch
      based on property searches in the class property table.

      This utility function can be used as a default way to return basic
      values (and eventually methods) from the property table if they doesn't
      need special management.

      Example:
      \code
      bool MyClass::getProperty( const String &key, Item &ret ) const
      {
         if ( key == "prop1" )
         {
            //...
         }
         else if ( key == "prop2" )
         {
            //...
         }
         else {
            return defaultProperty( key, ret );
         }

         return true; // we found it or we would have called defalutProperty.
      }
      \endcode
   */
   bool defaultProperty( const String &key, Item &prop ) const;

   /** Generates a property access error.
      If the property is in the class property table, a "read only" property error
      is raised, otherwise a "property not found" error is raised instead.

      Useful as a default terminator of simple subclass setProperty() overloads.
   */
   void readOnlyError( const String &key ) const;

protected:

   /** Return a default method among those recorded in the generator class.

      Takes the property table of the generator class and, if it's a function,
      methodizes it.

      @note This method is protected, and meant to extend objects by replacing
      directly the getProperty() method; external users should rely on
      getMethod().

      @param name The name of the method to be searched.
      @param mth The item that will be filled with a full ready method instance on success.
      @return true if the property exists and can be methodized.
   */
   bool getMethodDefault( const String &name, Item &mth ) const;

public:

   /** The base destructor does nothing.
      Use finalize() hook to dispose of internal, reflected data.
   */
   virtual ~CoreObject();

   /** Returns a valid sequence instance if this object's user data is a "Falcon Sequence".

      Sequences can be used in sequential operations as the for-in loops,
      or in functional sequence operations (as map, filter and so on).

      Objects containing a Falcon Sequence as user data can declare
      this changing this function and returning the sequence data.
   */
   Sequence* getSequence() const { return m_bIsSequence ? static_cast<Sequence*>( m_user_data ) : 0; }

   /** Returns a valid sequence instance if this object's user data is a "Falcon Data".

      Sequences can be used in sequential operations as the for-in loops,
      or in functional sequence operations (as map, filter and so on).

      Objects containing a Falcon Sequence as user data can declare
      this changing this function and returning the sequence data.
   */
   FalconData *getFalconData() const { return m_bIsFalconData ? static_cast<FalconData*>( m_user_data ) : 0; }

   /** Return the inner user data that is attached to this item. */
   void *getUserData() const { return m_user_data; }

   /** Set a generic user data for this object.
      This user data is completely un-handled by this class; it's handling
      is completely demanded to user-defined sub-classes and/or to property-level
      reflection system.
   */
   void setUserData( void *data ) { m_user_data = data; }

   /** Set a FalconData as the user data for this object.
      FalconData class present a minimal interface that cooperates with this class:
      - It has a virtual destructor, that is called when the wrapping CoreObject instance is destroyed.
      - It provides a clone() method that is called when the wrapping CoreObject is cloned.
      - It provides a gcMark() method, called when this Object is marked.
      - Serialization support is available but defaulted to fail.
   */
   void setUserData( FalconData* fdata );

   /** Set a Sequence as the user data for this object.
      Sequence class is derived from FalconData, and it adds an interface for serial
      access to items.
   */
   void setUserData( Sequence* sdata );

   /** Returns true if this object has the given class among its ancestors. */
   bool derivedFrom( const String &className ) const;

   /** Serializes this instance on a stream.
      \throw IOError in case of stream error.
   */
   virtual bool serialize( Stream *stream, bool bLive ) const;

   /** Deserializes the object from a stream.
      The object should be created shortly before this call, giving
      instruction to the constructor not to perform a full initialization,
      as the content of the object will be soon overwritten.

      Will throw in case of error.
      \throw IOError in case of stream error.
      \param stream The stream from which to read the object.
      \param bLive If true,
      \return External call indicator. In case it returns true, the caller
         should
   */
   virtual bool deserialize( Stream *stream, bool bLive );

   /** Performs GC marking of the inner object data.

     This marks properties and eventually inner data, if it's
     a FalconData* subclass instance.
   */
   virtual void gcMark( uint32 mark );

   /** Returns true if the class provides a certain property.
      Should not account Object metaclass properties, unless explicitly overloaded.
      \param prop The property to be searched.
      \return true if the generator class provides this property.
   */
   virtual bool hasProperty( const String &prop ) const;

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
   virtual CoreObject *clone() const = 0;

   /** Sets a property in the object.
      If the property is found, the value in the item is copied, otherwise the
      object is untouched and false is returned.

      In case of reflected objects, it may be impossible to set the property. In that case,
      the owning vm gets an error, and false is returned.

      \param prop The property to be set.
      \param value The item to be set in the property.
      \return ture if the property can be set, false otherwise.
   */
   virtual bool setProperty( const String &prop, const Item &value ) = 0;

   /** Stores an arbitrary string in a property.
      The string is copied in a garbageable string created by the object itself.
   */
   bool setProperty( const String &prop, const String &value );


   /** Returns the a shallow item copy of required property.
      The copy is shallow; strings, arrays and complex data inside
      the original property are shared.

      \param prop the property to be found
      \param value an item containing the object proerty copy.
      \return true if the property can be found, false otherwise
   */
   virtual bool getProperty( const String &prop, Item &value ) const = 0;

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
   bool getMethod( const String &propName, Item &method ) const
   {
      if ( getProperty( propName, method ) )
         return method.methodize( SafeItem(const_cast<CoreObject*>(this)) );
      return false;
   }

   /** Get the class that generated this object.
      This is the phisical core object instance that generated this object.
      The symbol in the returned class is the value returend by instanceOf().
      \return the CoreClass that were used to generate this object.
   */
   const CoreClass *generator() const { return m_generatedBy; }

   /** Apply the given values to this object.

      This function applies the values in dict
      inside this object, applying repeatedly them via setProperty.

      This function skips writeProperty and calls directly to the virtual
      setProperty method. This means that VM overrides are uncalled.

      If bRaiseOnError is set to true, a failed setProperty will
      cause the function to raise an AccessError. Otherwise, the
      function will ignore the error and proceed, but it will
      return false when completed.

      \param dict The property => value dictionary to be applied.
      \param bRaiseOnError if true, raise when a property in dict is not found here.
      \return true on success, false if some property was not found and
              bRaiseOnError is false.
   */
   bool apply( const ItemDict& dict, bool bRaiseOnError = false );

   /** Retrieves values of properties in this object.
      This method calls iteratively getProperty to fill the properties
      that corespond to the keys in dict.

      If bFillDict is true, the object is scanned and its properties
      are set in the dictionary instead.

      If bIgnoreMethods is true, method proeprties are skipped when
      bFillDict is true.

      If bRaiseOnError is true the function throws an AccessError when
      getProperty fails; this setting is ignored when bFillDict is true.

      \param dict The dictionary to be filled.
      \param bRaiseOnError If true, throw an AccessError when dict contains a key that
             doesn't match any property.
      \param bFillDict If true, ignore the contents of dict, and fill it with the
             actual properties instead.
      \param bIgnoreMethods When true, copy only data properties.
      \return True on success, false if bRaiseOnError is false and some keys in dict
              can't be found as properties of this object.
   */
   bool retrieve( ItemDict& dict, bool bRaiseOnError = false, bool bFillDict = false, bool bIgnoreMethods = true ) const;

   /** Override deep item accessor.
      By default, it searches a method named "getIndex__" and passes
      the target item to it; it raises if the method is not found.
   */
   virtual void readIndex( const Item &pos, Item &target );

   /** Override deep item accessor.
      By default, it searches a method named "setIndex__" and passes
      the target item to it; it raises if the method is not found.
   */
   virtual void writeIndex( const Item &pos, const Item &target );

   /** Override deep item accessor.
      Resolves into getProperty().
   */
   virtual void readProperty( const String &pos, Item &target );

   /** Override deep item accessor.
      Resolves into setProperty().
   */
   virtual void writeProperty( const String &pos, const Item &target );

   /** Enter a given state.

       If the state doesn't exist, will throw a CodeError (state not found) error.
       If it is found, it will change the state of the object via apply and will
       set the the current state to the given string.

       The function will also try to set the
       __leave -> setState -> __enter sequence.
    */
   void setState( const String& state, VMachine* vm );

   void setState( const String& state, ItemDict* stateDict );

   bool hasState() const { return m_state != 0; }
   const String& state() const { return *m_state; }

};

}

#endif

/* end of cobject.h */
