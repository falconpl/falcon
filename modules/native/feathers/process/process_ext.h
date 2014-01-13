/*
   FALCON - The Falcon Programming Language.
   FILE: process_ext.h

   Process module -- Falcon interface functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process module -- Falcon interface functions
   This is the module declaration file.
*/

#ifndef FALCON_FEATHERS_PROCESS_EXT_H
#define FALCON_FEATHERS_PROCESS_EXT_H

#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>
#include <falcon/classes/classshared.h>
#include <falcon/module.h>

namespace Falcon {
namespace Ext {

FALCON_DECLARE_FUNCTION( pid, "" );
FALCON_DECLARE_FUNCTION( tid, "" );
FALCON_DECLARE_FUNCTION( kill, "pid:N,severe:[B]" );
FALCON_DECLARE_FUNCTION( system, "command:S,background:[B]" );
FALCON_DECLARE_FUNCTION( systemCall, "command:S,background:[B],usePath:[B]" );
FALCON_DECLARE_FUNCTION( pread, "command:S,background:[B],grabAux:[B]" );
FALCON_DECLARE_FUNCTION( preadCall, "command:S,background:[B],grabAux:[B],usePath:[B]" );

class ClassProcess: public ClassShared
{
public:
   ClassProcess();
   virtual ~ClassProcess();

   virtual void* createInstance() const;
   virtual void* clone( void* source ) const;
   virtual void dispose( void* self ) const;
   virtual void describe( void* instance, String& target, int, int ) const;

   bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};


class ClassProcessEnum: public Class
{
public:
   ClassProcessEnum();
   virtual ~ClassProcessEnum();

   virtual void* createInstance() const;
   virtual void* clone( void* source ) const;
   virtual void dispose( void* self ) const;
   virtual void describe( void* instance, String& target, int, int ) const;

   bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
};


FALCON_DECLARE_ERROR(ProcessError);


class ProcessModule: public Module
{
public:
   ProcessModule();
   virtual ~ProcessModule();

   Class* classProcess() const { return m_classProcess; }

private:
   Class* m_classProcess;
};

}} // ns Falcon::Ext

#endif

/* end of process_ext.h */
