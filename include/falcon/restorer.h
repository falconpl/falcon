/*
   FALCON - The Falcon Programming Language.
   FILE: restorer.h

   Helper for cyclic joint structure deserialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 18 Oct 2011 17:45:15 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_RESTORER_H
#define FALCON_RESTORER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/pstep.h>

namespace Falcon {

class Stream;
class ModSpace;
class VMContext;
class Class;
class ModLoader;
class DataReader;

/** Helper for cyclic joint structure deserialization.
 
 \see Storer
*/

class FALCON_DYN_CLASS Restorer
{
public:
   Restorer();
   virtual ~Restorer();
   
   /** Restores the data in the required stream.
    \return true IF the restore was complete, false if the VMContext was altered.
    \throw IoError on error reading from the stream
    \thrown ParseError on semantic errors while reading from the stream.
    */
   virtual void restore( VMContext* vmc, Stream* rd, ModSpace* msp );
   
   virtual bool next( Class*& handler, void*& data, bool& first );
   virtual bool hasNext() const;
   virtual uint32 objCount() const;
   
   inline DataReader& reader() { return *m_reader; }

protected:
   Stream* m_stream;
   ModSpace* m_modspace;

private:
   class Private;
   Restorer::Private* _p;
   DataReader* m_reader;
   
   void readClassTable();
   void loadClasses( VMContext* ctx, ModSpace* msp );
   void readInstanceTable();
   
   bool readObjectTable( VMContext* ctx );
   bool unflatten( VMContext* ctx );

   
   class FALCON_DYN_CLASS ReadNext: public PStep
   {
   public:
      ReadNext(Restorer* owner): m_owner(owner) { apply = apply_; }
      virtual ~ReadNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      Restorer* m_owner; 
   };

   friend class ReadNext;
   ReadNext m_readNext;
   
   class FALCON_DYN_CLASS UnflattenNext: public PStep
   {
   public:
      UnflattenNext(Restorer* owner): m_owner(owner) { apply = apply_; }
      virtual ~UnflattenNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   
   private:
      Restorer* m_owner; 
   };

   friend class UnflattenNext;
   UnflattenNext m_unflattenNext;
  
   class FALCON_DYN_CLASS PStepLoadNextClass: public PStep
   {
   public:
      PStepLoadNextClass(Restorer* owner): m_owner(owner) { apply = apply_; }
      virtual ~PStepLoadNextClass() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      Restorer* m_owner; 
   };

   friend class PStepLoadNextClass;
   PStepLoadNextClass m_stepLoadNextClass;
};

}

#endif

/* end of deserializer.h */

