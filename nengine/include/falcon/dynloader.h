/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader.h

   Native shared object based module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DYNLOADER_H_
#define _FALCON_DYNLOADER_H_

#include <falcon/setup.h>

namespace Falcon
{

class Module;
class String;

/** Native shared object based module loader.
 */
class FALCON_DYN_CLASS DynLoader
{
public:
   DynLoader();
   virtual ~DynLoader();
   
   /** Loads a pre-compiled module from a data stream. 
    \param filePath The path where the shared object is stored.
    \param local_name The name under which the module is internally known.
    \return A valid module.
    \throw IOError on load error.
    
    \TODO Use a URI
    */
   Module* load( const String& filePath, const String& local_name );
   
   /** Returns a System-specific extension. */
   static const String& sysExtension();

private:

   Module* load_sys( const String& filePath );
};

}

#endif	/* _FALCON_DYNLOADER_H_ */

/* end of dynloader.h */
