/*
   FALCON - The Falcon Programming Language
   FILE: engstrings.cpp
   $Id: engstrings.cpp,v 1.16 2007/08/03 13:17:06 jonnymind Exp $

   Implementation of engine string table
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar feb 13 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Implementation of engine string table
*/

#include <falcon/engstrings.h>
#include <falcon/fassert.h>
#include <falcon/strtable.h>

namespace Falcon {

StringTable *engineStrings = 0;

char *en_table[] =
   {
      "The Falcon Programming Language",
      "Generic syntax error",
      "Incompatible unpack size for list assignment",
      "Break outside loops",
      "Continue outside loops",
      "Division by zero",
      "Module by zero",
      "Invalid operator",
      "Assignment to a constant",
      "Assignment to a non assignable symbol",
      //10
      "Repeated symbol in declaration",
      "Global statement not inside a function",
      "Symbol already defined",
      "Non callable symbol called",
      "Invalid comparation",
      "Symbol not defined in export directive",
      "Already exported all",
      "Misplaced state declaration",
      "Enter statement without function call in main body",
      "Leave statement in main body",
      //20
      "Static related instruction outside a function",
      "Object 'self' cannot be referenced from outside a class",
      "Object 'sender' cannot be referenced from outside a class",
      "Undefined symbol",
      "Invalid operands given opcode",
      ".local declared outside local context",
      "Directive .entry was already given",
      ".endfunc directive without corresponding .funcdef",
      "Directive .import outside function",
      "Directive .import references an undefined global symbol",
      //30
      "Too many local variables",
      "Too many parameters",
      "Switch directive was already pending",
      "Case directive is not in a switch",
      "Switch not open and .end_switch requested",
      "Directive .state out of function",
      "Property definition outside class",
      "Property definition after init definition",
      "Property already defined",
      "Too many properties defined for this class",
      //40
      "From clause already defined",
      "Too many inheritances for this class",
      "Invalid OPCODE",
      "Invalid operands",
      "Stack underflow",
      "Stack overflow",
      "Access array out of bounds",
      "No startup symbol found",
      "Explicitly raised item is uncaught",
      "System error in loading a binary module",
      "Binary module has not the 'falcon_module_init' startup procedure",
      "Module cannot be initialized",
      "Unrecognized module version",
      "Generic falcon module format error",
      "I/O error while loading a module",
      "Unclosed control structure",
      "Parse error at end of file",
      "Compiler is unprepared (use compile(Module *, Stream *)",
      "Label declared but never defined",
      "Requested property not found in object",
      "Deadlock detected",
      "Operator ''provides'' must be followed by a symbol name",
      "Duplicate case clause value",
      "Constructor already declared",
      "Static member initializers must be a constant expression",
      "Invalid string ID",
      "Class inhertits from a symbol that is not a class",
      "Trying to get a reference from something that's not a symbol",
      "State already defined",
      "Invalid SJMP (jump to state) in current context",
      "Symbol in HAS clause is not an attribute",
      "No internal class found for standalone object",
      "Statement pass used outside function",
      "Duplicate or clashing switch case",
      "Default block already defined in switch",
      "Invalid parameters in FOR related opcodes",
      "Variable is already declared global",
      "Service already published",
      "Required service has not been published",
      "Uncloneable object, as part of it is not available to VM",
      "Access to parameters outside function",
      "Can't create output file",
      "Mathematical domain error",
      "Invalid character while parsing source",
      "Closing a parenthesis, but never opened",
      "Closing square bracket, but never opened",
      "Invalid numeric format",
      "Invalid string escape sequence",
      "EOL in assembly string",
      "Invalid token",
      "Invalid directive",
      "Array byte access is read-only",
      "String too long for numeric conversion",
      "Invalid source data while converting to number",
      "Class specific declaration outside .class directive",
      "Bitwise operation on non-numeric parameters",
      "Syntax error in case statement",
      "Invalid statement in switch body",
      "Invalid statement in select body",
      "Syntax error in 'default' statement",
      "'end' statement without open contexts",
      "Syntax error in 'switch' statement",
      "Syntax error in 'select' statement",
      "Statement 'case' is valid only within switch or select statements",
      "Syntax error in 'load' directive",
      "Functions must be declared at toplevel",
      "Objects must be declared at toplevel",
      "Classes must be declared at toplevel",
      "Load directive must be called at toplevel",
      "Syntax error in 'while' statement",
      "Syntax error in 'if' statement",
      "Syntax error in 'else' statement",
      "Syntax error in 'elif' statement",
      "Syntax error in 'break' statement",
      "Syntax error in 'continue' statement",
      "Syntax error in 'for' statement",
      "Syntax error in 'forfirst' statement",
      "Syntax error in 'forlast' statement",
      "Syntax error in 'formiddle' statement",
      "Syntax error in 'try' statement",
      "Syntax error in 'catch' statement",
      "Syntax error in 'raise' statement",
      "Syntax error in function declaration",
      "Syntax error in 'static' statement",
      "Syntax error in 'state' statement",
      "Syntax error in 'launch' statement",
      "Syntax error in 'pass' statement",
      "Invalid value for constant declaration",
      "Syntax error in 'const' statement",
      "Syntax error in 'export' statement",
      "Syntax error in 'attributes' statement",
      "Argument for enter must be a variable",
      "Syntax error in 'enter' statement",
      "Syntax error in 'leave' statement",
      "Syntax error in 'for..in' statement",
      "Syntax error in 'pass..in' statement",
      "State leave value is not a valid expression",
      "Invalid attribute name",
      "Syntax error in 'class' statement",
      "Syntax error in 'has' definition",
      "Syntax error in 'object' statement",
      "Syntax error in 'global' statement",
      "Syntax error in 'return' statement",
      "Syntax error in array access",
      "Syntax error in function call",
      "Syntax error in 'lambda' statement",
      "Syntax error in '?:' expression",
      "Syntax error in dictionary declaration",
      "Syntax error in array declaration",
      "Syntax error in 'give' statement",
      "Syntax error in 'def' statement",
      "Syntax error in 'for.' statement",
      "Syntax error in fast print statement",
      "Syntax error in directive",
      "Syntax error in import statement",

      "New line in literal string",
      "Duplicate type identifier in catch selector",
      "Default catch block already defined",
      "Format not applicable to object",
      "Block 'forfirst' already declared",
      "Block 'forlast' already declared",
      "Block 'formiddle' already declared",
      "Statement '.=' must be inside a for/in loop",
      "VM received a suspension request in an atomic operation",
      "Access to private member not through 'self'",
      "Unbalanced parenthesis at end of file",
      "Unbalanced square parenthesis at end of file",
      "Unclosed string at end of file",
      "Unknown directive",
      "Invalid value for directive",

      "Can't open file",
      "Load error",
      "File not found",
      "Invalid or damaged Falcon VM file",
      "Operation not supported by the module loader",

      "Invalid parameters",
      "Mandatory parameter missing",
      "Invalid parameters type",
      "Parameters content invalid/out of range",
      "Parse error in indirect code",
      "Parse error in expanded string",
      "Parse error in format specifier",

      "Unrecognized error code",

      "requres an object and a string representing a property, plus the new value",
      "parameter array contains non string elements" ,
      0
   };


const String &getMessage( uint32 id )
{
   if ( engineStrings == 0 )
      setEngineLanguage( "C" );

   const String *data = engineStrings->get( id );
   fassert( data != 0 );
   return *data;
}

bool setTable( StringTable *tab )
{
   if ( engineStrings == 0 || engineStrings->size() == tab->size() )
   {
      engineStrings = tab;
      return true;
   }
   return false;
}

bool setEngineLanguage( const String &language )
{
   if ( language == "en-US" || language == "en-GB" || language == "C" )
   {
      delete engineStrings;
      engineStrings = new StringTable;
      engineStrings->build( en_table );
      return true;
   }

   // residual criterion: using english but...
   delete engineStrings;
   engineStrings = new StringTable;
   engineStrings->build( en_table );

   // ... signal that we didn't found the language.
   return false;
}



}

/* end of engstrings.cpp */
