/*
   FALCON - The Falcon Programming Language.
   FILE: compiler.h

   Falcon source compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Feb 2011 16:41:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_COMPILER_H
#define FALCON_COMPILER_H

#include <falcon/vm.h>

namespace Falcon {

class Stream;

class Compiler
{
public:
   Compiler();
   virtual ~Compiler();

   Module* compile( Stream* in );

   
private:
   VMachine m_vm;

};

}

#endif