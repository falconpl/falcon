/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_vm.cpp

   Falcon module for Apache 2

   Specialized Virtual Machine with watchdogs
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Mar 2012 16:59:57 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#include <stdio.h>
#include "mod_falcon_vm.h"

ApacheVMachine::ApacheVMachine():
   m_terminateRequest(false)
{}

ApacheVMachine::~ApacheVMachine()
{
}
   
 void ApacheVMachine::terminate()
 {
    m_termMtx.lock();
    m_terminateRequest = true;
    m_termMtx.unlock();
 }

 void ApacheVMachine::periodicCallback()
 {
    m_termMtx.lock();
    bool bTerm = m_terminateRequest;
    m_termMtx.unlock();
    
    fprintf( stderr, "Callback called now...\n" );
    
    if( bTerm )
    {
       throw new Falcon::InterruptedError( Falcon::ErrorParam( Falcon::e_interrupted, __LINE__ )
          .extra("Interrupted due timeout") );
    }
 }
