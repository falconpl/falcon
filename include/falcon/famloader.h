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

class Stream;
class Module;
class String;
class ModSpace;

/** Precompiled module deserializer.
 */
class FALCON_DYN_CLASS FAMLoader
{
public:
   FAMLoader( ModSpace* ms );
   virtual ~FAMLoader();
   
   /** Loads a pre-compiled module from a data stream. 
    \param r The reader where the binary module is stored.
    \param local_name The name under which the module is internally known.
    */
   Module* load( Stream* r, const String& uri, const String& local_name );

   /** Module space bound with this fam loader. */
   ModSpace* modSpace() const { return m_modSpace; }
private:
   ModSpace* m_modSpace;
};

}

#endif	/* _FALCON_FAMLOADER_H_ */

/* end of famloader.h */
