/*
   FALCON - The Falcon Programming Language.
   FILE: loaderprocess.h

   VM Process specialized in loading modules into the VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Dec 2012 19:52:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_LOADERPROCESS_H_
#define _FALCON_LOADERPROCESS_H_

#include <falcon/setup.h>
#include <falcon/process.h>
#include <falcon/pstep.h>

namespace Falcon {

class ModSpace;
class Module;
class VMContext;
class SynFunc;

/** VM Process specialized in loading modules into the VM.
 * \note THIS IS A TODO / work in progress. Do not use
 * for production.
 */
class FALCON_DYN_CLASS LoaderProcess: public Process
{
public:
   LoaderProcess( VMachine* vm, ModSpace* ms );

   void loadModule( const String& modUri, bool isUri, bool launchMain=false );

   ModSpace* modSpace() const { return m_ms; }

   Module* mainModule() const;

   void setMainModule( Module* mod );

protected:
   virtual ~LoaderProcess();

private:
   ModSpace* m_ms;
   Module* m_mainModule;
   SynFunc* m_loaderEntry;

   class FALCON_DYN_CLASS PStepSetupMain: public PStep
   {
   public:
      PStepSetupMain( LoaderProcess* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepSetupMain() {};
      virtual void describeTo( String& str ) const { str = "PStepSetupMain"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      LoaderProcess* m_owner;
   };
   PStepSetupMain m_stepSetupMain;

};

}

#endif /* LOADERPROCESS_H_ */

/* end of loaderpocess.h */

