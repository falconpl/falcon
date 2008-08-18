/*
   FALCON - The Falcon Programming Language.
   FILE: intcomp.h

   Complete encapsulation of an incremental interactive compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Aug 2008 11:10:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_INTCOMP_H
#define FALCON_INTCOMP_H

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/module.h>
#include <falcon/vm.h>

namespace Falcon
{

class FlcLoader;

/** Interactive compiler.
   This compiler is meant to incrementally compile, load dependencies and
   execute on the fly single statements.

   For this reason, the compiler is provided with a VM and a flexy module;
   the compiler independently creates the module (which may be referenced
   and taken also externally) and exposes a function that allows incremental
   compilation. Using compileNext(), this class independently loads dependencies
   as they are found, executes statements and fills the module symbol table.

   The compiler may be provided with a VM generated from the outside, or it
   will create a standard VM on its own (which can be configured at a later
   moment.
*/
class InteractiveCompiler: public Compiler
{
   VMachine *m_vm;
   LiveModule *m_lmodule;
   FlcLoader *m_loader;
   void loadNow( const String &name, bool isFilename );

public:

   /** Create the interactive compiler.
      If a VM is not provided, it will be automatically created.
      Notice that the compiler will apply its error handler to the loader
      at compile time.
   */
   InteractiveCompiler( FlcLoader *loader, VMachine *vm=0 );
   ~InteractiveCompiler();

   typedef enum {
      e_nothing,
      e_more,
      e_incomplete,
      e_decl,
      e_statement,
      e_expression,
      e_call,
      e_error,
      e_vm_error
   } t_ret_type;

   /** Compile another code slice coming from the stream.
      The calling application will receive the control back when the compilation,
      and eventually the execution of the needed steps are completed.

      The return value may be one of the t_ret_type enumeration, and the
      calling application can take proper actions.

      A return indicating error won't block the compiler nor invalidate, which is still
      available to compile incrementally other code slices.
   */
   t_ret_type compileNext( Stream *input );

   /** Compile another codeslice from a string.
      \see compileNext( Stream *);
   */
   t_ret_type compileNext( const String &input );

   VMachine *vm() const { return m_vm; }

   virtual void addLoad( const String &name, bool isFilename );
   virtual void addNamespace( const String &nspace, const String &alias, bool full=false, bool filename=false );


   FlcLoader *loader() const { return m_loader; }
   void loader( FlcLoader *l ) { m_loader = l; }
};

}

#endif

/* end of intcomp.h */
