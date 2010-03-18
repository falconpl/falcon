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
#include <falcon/intcomp.h>

namespace Falcon
{

class ModuleLoader;

/** Interactive compiler.
   This compiler performs incremental compiling, loading dependencies and
   can also execute on the fly single statements.

   For this reason, the compiler is provided with a VM and a flexy module;
   the compiler independently creates the module (which may be referenced
   and taken also externally) and exposes a function that allows incremental
   compilation. Using compileNext(), this class independently loads dependencies
   as they are found, executes statements and fills the module symbol table.

   The compiler may be provided with a VM generated from the outside, or it
   will create a standard VM on its own (which can be configured at a later
   time.
*/
class FALCON_DYN_CLASS InteractiveCompiler: public Compiler
{
   VMachine *m_vm;
   LiveModule *m_lmodule;
   ModuleLoader *m_loader;
   bool m_interactive;

   void loadNow( const String &name, bool isFilename, bool bPrivate );

public:

   /** Create the interactive compiler.
      If a VM is not provided, it will be automatically created.
      Notice that the compiler will apply its error handler to the loader
      at compile time.
   */
   InteractiveCompiler( ModuleLoader *loader, VMachine *vm );
   ~InteractiveCompiler();

   typedef enum {
      /** Do-nothing statements (comments, whitespaces ...)*/
      e_nothing,
      /** We need more to finish the current statement */
      e_more,
      /** Incomplete context */
      e_incomplete,
      /** Statement was a complete declaration (function, object, class...) */
      e_decl,
      /** Normal statement (assignment, if block, while block...) */
      e_statement,
      /** Stand alone expression (sum, sub, single value) */
      e_expression,
      /** Special expression consisting of a single call to a sub expression.
         Usually, the user expects a return value.
      */
      e_call,
      
      e_terminated
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

   t_ret_type compileAll( const String &input );

   VMachine *vm() const { return m_vm; }

   virtual void addLoad( const String &name, bool isFilename );
   virtual void addNamespace( const String &nspace, const String &alias, bool full=false, bool filename=false );


   ModuleLoader *loader() const { return m_loader; }
   void loader( ModuleLoader *l ) { m_loader = l; }


   bool interactive() const { return m_interactive; }
   /** Sets or resets interactive mode.
      In interactive mode, the compiler accepts one statement at a time and transforms
      expressions into "return [expression]" statements to allow their value
      to be visible after the execution.

      Also, the lexer is configured so that it can accept partial elements from
      streams and tell back about a partial status to the compiler.
   */
   void interactive( bool iactive ) { m_interactive = iactive; }
};

}

#endif

/* end of intcomp.h */
