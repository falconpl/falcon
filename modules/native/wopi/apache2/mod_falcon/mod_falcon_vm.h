/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_vm.h

   Falcon module for Apache 2

   Specialized Virtual Machine with watchdogs
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Mar 2012 16:59:57 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef MOD_FALCON_VM_H
#define MOD_FALCON_VM_H

#include <falcon/vm.h>

class ApacheVMachine: public Falcon::VMachine
{
public:
   ApacheVMachine();
   virtual ~ApacheVMachine();
   
   void terminate();
   virtual void periodicCallback();

private:
   bool m_terminateRequest;
   Falcon::Mutex m_termMtx;
};

#endif
