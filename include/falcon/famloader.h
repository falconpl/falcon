/*
   FALCON - The Falcon Programming Language.
   FILE: famloader.h

   Precompiled module deserializer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FAMLOADER_H_
#define _FALCON_FAMLOADER_H_

#include <falcon/setup.h>
#include <falcon/restorer.h>

namespace Falcon
{

class Stream;
class Module;
class String;
class ModSpace;
class VMContext;

/** Precompiled module deserializer.
 */
class FALCON_DYN_CLASS FAMLoader
{
public:
   FAMLoader( ModSpace* ms );
   virtual ~FAMLoader();

   /** Loads a pre-compiled module from a data stream.
    \param input The reader where the binary module is stored.
    \param uri The URI where the module is being read from.
    \param local_name The name under which the module is internally known.

    \note The stream \b input is sent to the garbage collector. The ownership passes
    to the Falcon virtual machine.
    */
   void load( VMContext* ctx, Stream* input, const String& uri, const String& local_name );

   /** Module space bound with this fam loader. */
   ModSpace* modSpace() const { return m_modSpace; }
private:
   ModSpace* m_modSpace;

   class FALCON_DYN_CLASS PStepLoad: public PStep
   {
   public:
      PStepLoad( FAMLoader* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepLoad() {};
      virtual void describeTo( String& str ) const { str = "PStepLoad"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      FAMLoader* m_owner;
   };
   PStepLoad m_stepLoad;

   friend class PStepLoad;
};

}

#endif	/* _FALCON_FAMLOADER_H_ */

/* end of famloader.h */
