/*
   FALCON - The Falcon Programming Language.
   FILE: storer.h

   Helper for cyclic joint structure serialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 18 Oct 2011 17:45:15 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STORER_H
#define FALCON_STORER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/pstep.h>

namespace Falcon {

class Class;
class DataWriter;
class VMContext;

/** Helper for cyclic joint structure serialization.
 
 The storer role is that to create a coherent
 output table where all the needed classes are referenced
 and all the cycles are resolved and reproduced at
 de-serialization.
 
 The storer receives one or more object/class pairs from
 the engine through the store() method. For each object, 
 it records the class it is associated to it and the dependencies 
 with other objects. For instance, if an array of item is to be stored, each
 item in the array is stored as well. Then, all the stored entities are
 delivered to a stream through the commit() method.
 
 The Class::flatten mechanism is provided and called back so
 that a class can decide which other stand-alone objects its
 managed type depends on. For instance, the traverse mechanism
 of an array would try to store all the items in the array. Then,
 each object is saved on the stream through Class::store.
 
 WObjects are saved by reference, and multiply stored objects
 are actually stored just once, with their identifier in the
 storage saved and proposed to the Class::unflatten 
 method during deserialization. 
 
 In practice, during the deserialization process first all the objects 
 stored in the storer are 
 reconstructed from their data in the stream through Class::restore,
 then each forming object is proposed to Class::unflatten to give it
 the chance to attach other entities that the object is depending on. 
 
 The storer records full information about the classes the object depends on,
 so that is possible for the engine to find those classes or try to load them
 on the fly during the de-serialization process.
 
 All the mechanism is integrated with the VM: if a class needs some script
 element to interact with the serialization process (in any of the 4 phases,
 flatten, store, restore, unflatten), the process is suspended and the VM can
 punch in to execute PSteps. In case of FalconClass instances, overriding
 this methods through falcon methods named as flatten, store, restore or unflatten
 will have the effect of allowing the classes to override their default behavior
 and provide personalized serialization strategies.
 
 @section storer_storage_format Storage format
 
 The storer writes the data it is given to a stream in the following format:
 - Class Table.
    -- Number of classes (int32)
    -- For each class:
      --- Class name.
      --- Full name and then URI of the module, or empty string if none
 - Instance table.
   -- Number of entities to be deserialized (int32)
   -- For each item:
      --- The ID in the instance (and same ID in object table) of the item
         that was stored at this point (int32).
 - Object table.
    -- Number of entities to be deserialized (int32)
    -- For each entity:
       --- A synchronizer marker stored as int32: 0xFECDBA98      
       --- Sequential ID of the class handling the stored object (int32).
       --- Number of dependencies to be unflattened (int32, possibly 0).
       --- For each dependency:
          ---- Sequential ID of the object on which the instance depends.
       --- The raw data obtained through Class::store for this item.

 Integers and strings in the Class and Instance tables are stored through
 the standard DataWriter::write methods, so they are to be restored similarly
 through DataReader methods. The Object ID is simply the progressive number
 of the object as stored in the Object Table, 0-based. Similarly, the Class ID
 is simply the progressive number of storage of a class in the Class table, 
 0-based.
 
 @section storer_usage_patterns Usage patterns
 
 The storer class is provided to minimize the requirements and centralize the
 management of the serialization of deep and convoluted interrelated data. 
 
 A single store() operation in the Storer will match a single restore() 
 operation in the Restorer. 
 
 A storer can be used to commit multiple different stored data
 to even different streams, or commit multiple times the same data to the same
 stream.
 
 Each commit() will be "atomically" stored on the stream, and benerates an unbreakable
 data unit that must be loaded all at once by the restorer. This means that a
 restorer must deserialize all the items that have been stored by a single commit()
 before returning the control to the calling program.
 In other words, deserialization is not lazy in absolute, but it can be considered
 lazy across different commit() points.
 
 Consider the following code, expressed for simplicity in Falcon code (which
 relects the same C++ classes that are documented here):
 
 @code
   import from vfs in vfs
   file = vfs.create( "storetest"  )
   storer = Storer()

   storer.store( 1 )
   storer.store( 2 )
   storer.store( "Hello" )
   storer.store( "World" )
   storer.commit( file )

   file.close()

   \/\* ... *\/
   file = vfs.open( "storetest"  )
   restorer = Restorer()
   restorer.restore( file )

   > restorer.next()      // 1
   > restorer.next()      // 2
   > restorer.next()      // "Hello"
   > restorer.next()      // "World"
   > restorer.next()      // throws an exception.
 @endcode
 
 In this code, all the 4 stored objects are restored at <b>restorer.restore</b>,
 while subsequent calls to <b>restorer.next</b> will just return objects that
 have already been fully created and are already alive in <b>restorer</b>.
 
 Complex programs willing to partition their storage may opt for a strategy
 of performing multiple commits, while programs knowing that the data set
 must either be fully loaded or fail altogether can save their whole dataset
 through a single commit.
 
 @note Storer::commit stores the class table and the instance table on the stream
 before attempting the first serialization through Class::store. This means that
 an error in the serialization (e.g. an exception thrown by some Falcon level
 override of Class::store) will leave a partial output on the stream. If this is
 undesired, it is advisable to use a memory stream before sending data on
 the network or on a file that should either be valid or not be generated, or
 alteratively use a temporary file and then renaming it or streaming it on
 the network when done.
 
 @note If the same instance of an object is stored more than once, the same
 instance is returned more than once in by Restorer::restore(). Similarly,
 if two different deeply flattened objects reference the same instance,
 they will still reference the same instance when restored.
 
 @note It is not possible to store references. In that case, the referenced
 item is stored, and at restore the reference data is lost.
 */

class FALCON_DYN_CLASS Storer
{
public:
   Storer( VMContext* ctx );
   virtual ~Storer();
   
   /** Stores an objet.
    \return true if the operation was completed, false if a class requested
    VM processing.
    
    */
   virtual bool store( Class* handler, void* data );
   
   /** Writes the stored objects on the stream.
    @param wr The data stream on which the item should be stored.
    \return true if the operation was completed, false if a class requested
    VM processing. 
    */
   virtual bool commit( Stream* dataStream );
   
private:      
   class Private;
   Private* _p;
   
   VMContext* m_ctx;
   // internally used during serialization.
   DataWriter* m_writer;
      
   // Using void* because we'll be using private data for that.
   bool traverse( Class* handler, void* data, bool isTopLevel = false, void** objd = 0 );
   void writeClassTable( DataWriter* wr );
   void writeInstanceTable( DataWriter* wr );
   bool writeObjectTable( DataWriter* wr );
   void writeObjectDeps( uint32 pos, DataWriter* wr );
   bool writeObject( VMContext* ctx, uint32 pos, DataWriter* wr );
   
   class FALCON_DYN_CLASS TraverseNext: public PStep
   {
   public:
      TraverseNext(Storer* owner): m_owner(owner) { apply = apply_; }
      virtual ~TraverseNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   
   private:
      Storer* m_owner; 
   };

   friend class TraverseNext;
   TraverseNext m_traverseNext;
   
   class FALCON_DYN_CLASS WriteNext: public PStep
   {
   public:
      WriteNext(Storer* owner): m_owner(owner) { apply = apply_; }
      virtual ~WriteNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      Storer* m_owner; 
   };

   friend class WriteNext;
   WriteNext m_writeNext;
   
};

}

#endif

/* end of storer.h */
