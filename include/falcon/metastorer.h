/*
   FALCON - The Falcon Programming Language.
   FILE: metastorer.h

   Storer for classes and engine-opaque elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jan 2012 21:41:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_METASTORER_H_
#define _FALCON_METASTORER_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {

class VMContext;
class DataWriter;
class DataReader;
class ItemArray;

/** Storer for classes and engine-opaque elements.
 
 Storage of items known by the Falcon engine is performed by Class
 subclasses.
 
 However, classes themselves might be subject to storage. Also, there might
 be elements that are opaquely handled by a single class but are internally
 represented by a class hierarcy. For instance, the Function class handles
 a series of different entitiees that are all exposed to the engine as
 functions, but might be internally quite different.
 
 To cover the need of correctly storing those entities as well, this MetaStorer
 is provided.
 
 The engine can register meta storers under a specific name. The Class handlers 
 needing to construct dynamic instances out of an opaque internal class
 hierarcy can store the name of the MetaStore instead, and ask it back to
 the engine. 
 
 The most important meta-stored classes (as several dynamic class types,
 like FalconClass, HyperClass, Prototype and FlexyClass) are known to the
 engine in its initialization. New entities can be posted by third party 
 modules as they get loaded in the engine.
 
 */
class MetaStorer
{
public:
   /** Creates the storer with a name.
    \param name The name of the entity that is handled by this. 
    
    The name should be unique in the system, or at least, unique
    to the metastorers someone might want to register. 
    
    It's advisable to use this naming scheme:    
    "$[.ModName[.SubModName]].EntityName";
    */
   MetaStorer( const String& name ):
      m_name(name)
   {}
   
   virtual ~MetaStorer() {}
   
   /** Returns the name of this meta storer. */
   const String& name() const { return m_name; }
   
   /** Store an instance to a determined storage media.
    @param ctx A virtual machine context where the serialization occours.
    @param stream The data writer where the instance is being stored.
    @param instance The instance that must be serialized.
    @throw IoError on i/o error during serialization.
    @throw UnserializableError if the class doesn't provide a class-specific
    serialization.

    By default, the base class raises a UnserializableError, indicating that
    there aren't enough information to store the live item on the stream.
    Subclasses must reimplement this method doing something sensible.

    @see class_serialize
    */
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const = 0;

   /** Restores an instance previously stored on a stream.
    \param ctx A virtual machine context where the deserialization occours.
    \param stream The data writer where the instance is being stored.
    \param empty A pointer that will be filled with the new instance (but see below)
    \throw IoError on i/o error during serialization.
    \throw UnserializableError if the class doesn't provide a class-specific
    serialization.

    By default, the base class raises a UnserializableError, indicating that
    there aren't enough information to store the live item on the stream.
    Subclasses must reimplement this method doing something sensible.

    @note The \b empty pointer will receive the newly created and deserialized instance,
    but flat classes (those for which isFlatInstance() returns true) expect this
    pointer to be preallocated as an entity of class Item.

    \see class_serialize
   */
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const = 0;

   /** Called berfore storage to declare some other items that should be serialized.
    \param ctx A virtual machine context where the deserialization occours.
    \param stream The data writer where the instance is being stored.
    \param instance The instance that must be serialized.

    This method is invoked before calling store(), to give a change to the class
    to declare some other items on which this instance is dependent.

    The subclasses should just fill the subItems array or eventually invoke the
    VM passing the subItem array to the called subroutine. The items that
    are stored in the array will be serialized afterwards.

    The subItems array is garbage-locked by the Serializer instance that is
    controlling the serialization process.

    The base class does nothing.

    \see class_serialize
   */
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const = 0;

   /** Called after deserialization to restore other items.
    \param ctx A virtual machine context where the deserialization occours.
    \param stream The data writer where the instance is being stored.
    \param instance The instance that must be serialized.

    This method is invoked after calling restore(). The subItem array is
    filled with the same items, already deserialized, as it was filled by
    flatten() before serialization occured.

    The subItems array is garbage-locked by the Deserializer instance that is
    controlling the serialization process.

    The base class does nothing.

    \see class_serialize
   */
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const = 0;
   
private:
   String m_name;
};

}

#endif	/* _FALCON_METASTORER_H_ */

/* end of metastorer.h */
