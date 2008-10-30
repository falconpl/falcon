/*
   The Falcon Programming Language
   FILE: dynlib_mod.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Internal logic functions - implementation.
*/

#include "dynlib_mod.h"

namespace Falcon {

FunctionAddress::~FunctionAddress()
{
   // nothing needed to be done.
}

void FunctionAddress::gcMark( VMachine * )
{
   // nothing to mark
}

FalconData *FunctionAddress::clone() const
{
   return new FunctionAddress( *this );
}


void FunctionAddress::call( VMachine *vm, int32 firstParam ) const
{
}



bool DynFuncManager::isFalconData() const
{
   return true;
}

void *DynFuncManager::onInit( Falcon::VMachine * )
{
   return 0;
}

void DynFuncManager::onDestroy( Falcon::VMachine *, void *user_data )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>( user_data );
   delete fa;
}

void *DynFuncManager::onClone( Falcon::VMachine *, void *user_data )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>( user_data );
   return fa->clone();
}


}


/* end of dynlib_mod.cpp */
