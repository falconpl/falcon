/*
   FALCON - The Falcon Programming Language.
   FILE: testsuite.h

   Helper functions allowing the application to inteact with the testsuite module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Helper functions allowing the application to inteact with the testsuite module.
*/

#ifndef flc_testsuite_H
#define flc_testsuite_H

#include <falcon/string.h>

namespace Falcon {

namespace TestSuite {

   /** Check wether the last module execution exited with a succes or a failure.
      If the last module that was passed into the VM called success() to terminate
      it's execution, this function returns true. If it called failure(), this function
      return false.

      In case the script termiantes without calling success or failure, the return is
      undetermined. For this reason, it's advisable to reset the success status with
      setSuccesss() function to a consistent value.

      \return true if the last module execution was succesful
   */
   bool getSuccess();

   void setSuccess( bool mode );

   /** Retreives the last failure reason.
      If a module caused a failure, this function returns a string that the module
      passed to the failure() falcon function. If the module didn't set a reason
      for it's failure, the returned string is empty.
   */
   const ::Falcon::String &getFailureReason();


   /** Sets the name of the test.
      Used for internal reporting and alive functions.
      \param name the name of the test that is being run.
   */
   void setTestName( const ::Falcon::String &name );

   /** Get internal test timings.
   */
   void getTimings( ::Falcon::numeric &totTime, ::Falcon::numeric &numOps );

   /** Get internal test time factor.
   */
   ::Falcon::int64 getTimeFactor();

   /** set internal test time factor.
   */
   void setTimeFactor( ::Falcon::int64 factor );
}
}

#ifdef FALCON_EMBED_MODULES
/** Embedding function.
   This function can be directly called by embedder application to create the
   testsuite module.

   This function is defined only if the symbol FALCON_EMBED_MODULES is defined;
   otherwise, the standard dynamic lynk library initialization function is declared,
   and the embedder would have to use the module loader to bring the modules in
   the application.

   With FALCON_EMBED_MODULES, the embedder application can link the desidred modules
   directly from soruces and then call the function that prepares the module data.
*/
Falcon::Module *init_testsuite_module();
#endif

#endif

/* end of testsuite.h */
