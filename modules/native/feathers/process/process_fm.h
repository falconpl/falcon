/*
   FALCON - The Falcon Programming Language.
   FILE: process_fm.h

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

#ifndef FALCON_FEATHERS_PROCESS_FM_H
#define FALCON_FEATHERS_PROCESS_FM_H

#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>
#include <falcon/classes/classshared.h>
#include <falcon/module.h>

#define FALCON_FEATHER_PROCESS_NAME "process"

namespace Falcon {
namespace Feathers {

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


class ModuleProcess: public Module
{
public:
   ModuleProcess();
   virtual ~ModuleProcess();

   Class* classProcess() const { return m_classProcess; }

private:
   Class* m_classProcess;
};

}} // ns Falcon::Feathers

#endif

/* end of process_fm.h */
