/*
   FALCON - The Falcon Programming Language.
   FILE: print.h

   Falcon core module -- print/printl functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 11:54:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_PRINT_H
#define	FALCON_CORE_PRINT_H

#include <falcon/function.h>
#include <falcon/pstep.h>
#include <falcon/string.h>

#include <falcon/fassert.h>

#include <falcon/trace.h>

namespace Falcon {
namespace Ext {

/*#
   @function print
   @inset core_basic_io
   @param ... An arbitrary list of parameters.
   @brief Prints the contents of various items to the standard output stream.

   This function is the default way for a script to say something to the outer
   world. Scripts can expect print to do a consistent thing with respect to the
   environment they work in; stand alone scripts will have the printed data
   to be represented on the VM output stream. The stream can be overloaded to
   provide application supported output; by default it just passes any write to
   the process output stream.

   The items passed to print are just printed one after another, with no separation.
   After print return, the standard output stream is flushed and the cursor (if present)
   is moved past the last character printed. The function @a printl must be used,
   or a newline character must be explicitly placed among the output items.

   The print function has no support for pretty print (i.e. numeric formatting, space
   padding and so on). Also, it does NOT automatically call the toString() method of objects.

   @see printl
*/


class FALCON_DYN_CLASS FuncPrintBase: public Function
{
public:   
   FuncPrintBase(const String& name, bool ispl );

   virtual ~FuncPrintBase();

   virtual void apply( VMContext* ctx, int32 nParams );

private:
   class NextStep: public PStep
   {
   public:
      NextStep();
      static void apply_( const PStep* ps, VMContext* ctx );
      void printNext( VMContext* ctx, int count ) const;
      bool m_isPrintl;
   };

   NextStep m_nextStep;
};


/*#
   @function printl
   @inset core_basic_io
   @param ... An arbitrary list of parameters.
   @brief Prints the contents of various items to the VM standard output stream, and adds a newline.

   This functions works exactly as @a print, but it adds a textual "new line" after all the
   items are printed. The actual character sequence may vary depending on the underlying system.

   @see print
*/

/**
 Class implementing the standard printl function.
 */
class FALCON_DYN_CLASS FuncPrintl: public FuncPrintBase
{
public:
   FuncPrintl():
      FuncPrintBase( "printl", true )
      {}

   virtual ~FuncPrintl() {}
};

/**
 Class implementing the standard printl function.
 */
class FALCON_DYN_CLASS FuncPrint: public FuncPrintBase
{
public:
   FuncPrint():
      FuncPrintBase( "print", false )
      {}

   virtual ~FuncPrint() {}
};

}
}

#endif	/* PRINT_H */
