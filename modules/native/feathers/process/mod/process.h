/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.h

   Process API definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process API definitions
*/

#ifndef flc_process_mod_H
#define flc_process_mod_H

#include <falcon/genericvector.h>
#include <falcon/cacheobject.h>
#include "../sys/process.h"

namespace Falcon { namespace Mod {

class ProcessEnum : public CacheObject
{
public:
   ProcessEnum(CoreClass const* cls);
   virtual ~ProcessEnum();

   // not cloneable
   ProcessEnum *clone() const { return 0; }
   Sys::ProcessEnum* handle();

private:
   class Impl;
   Impl* m_impl;
};

class Process : public CacheObject
{
public:
   Process(CoreClass const* cls);
   virtual ~Process();

   // not cloneable
   Process *clone() const { return 0; }
   Sys::Process* handle();


private:
   class Impl;
   Impl* m_impl;
};

/**  Tokenizes a command string with its paremters and appends it to a vector.
 * \param argv Command tokens will be appended to it.
 * \param params Command string.
 */
 void argvize(GenericVector& argv, const String &params);

const char *shellName();
const char *shellParam();

}} // ns Falcon::Mod

#endif

/* end of process_mod.h */
