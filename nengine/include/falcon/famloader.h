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

namespace Falcon
{

class DataReader;
class Module;
class String;

/** Precompiled module deserializer.
 */
class FALCON_DYN_CLASS FAMLoader
{
public:
   FAMLoader();
   virtual ~FAMLoader();
   
   /** Loads a pre-compiled module from a data stream. 
    \param r The reader where the binary module is stored.
    \param local_name The name under which the module is internally known.
    */
   Module* load( DataReader* r, const String& uri, const String& local_name );
};

}

#endif	/* _FALCON_FAMLOADER_H_ */

/* end of famloader.h */
