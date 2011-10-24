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
   
   /** Writes the stored objects on the stream
    \return true if the operation was completed, false if a class requested
    VM processing. 
    */
   virtual bool commit( DataWriter* wr );
   
private:      
   class Private;
   Private* _p;
   
   VMContext* m_ctx;
   // internally used during serialization.
   DataWriter* m_writer;
      
   // Using void* because we'll be using private data for that.
   bool traverse( Class* handler, void* data );
   void writeClassDict( DataWriter* wr );
   bool doSerialize( DataWriter* wr );
   bool doSerializeItems( uint32 pos, DataWriter* wr );
   
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
   
   class FALCON_DYN_CLASS WriteNextPart: public PStep
   {
   public:
      WriteNextPart(Storer* owner): m_owner(owner) { apply = apply_; }
      virtual ~WriteNextPart() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      Storer* m_owner; 
   };

   friend class WriteNextPart;
   WriteNextPart m_writeNextPart;
};

}

#endif

/* end of storer.h */
