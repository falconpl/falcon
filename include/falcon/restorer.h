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

/** Helper for cyclic joint structure deserialization.
 
 \see Storer
*/

class FALCON_DYN_CLASS Restorer
{
public:
   Restorer();
   Restorer( VMContext* vmc );
   virtual ~Restorer();
   
   /** Restores the data in the requried stream.
    \return true IF the restore was complete, false if the VMContext was altered.
    \throw IoError on error reading from the stream
    \thrown ParseError on semantic errors while reading from the stream.
    */
   virtual bool restore( Stream* rd, ModSpace* msp=0, ModLoader* ml=0 );
   
   virtual bool next( Class*& handler, void*& data, bool& first );
   virtual bool hasNext() const;
   virtual uint32 objCount() const;
   
   void context( VMContext* vmc ) { m_ctx = vmc; }
   VMContext* context() const { return m_ctx; }
   
private:
   class Private;
   Private* _p;
   
   VMContext* m_ctx;
   
   void readClassTable();
   bool loadClasses( ModSpace* msp, ModLoader* ml );
   void readInstanceTable();
   
   bool readObjectTable();   
   bool unflatten();

   
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
  
   class FALCON_DYN_CLASS LinkNext: public PStep
   {
   public:
      LinkNext(Restorer* owner): m_owner(owner) { apply = apply_; }
      virtual ~LinkNext() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   private:
      Restorer* m_owner; 
   };

   friend class LinkNext;
   LinkNext m_linkNext;
};

}

#endif

/* end of deserializer.h */

