/*
   FALCON - The Falcon Programming Language.
   FILE: comp.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FASM_COMPILER_H
#define FASM_COMPILER_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/fstream.h>
#include <falcon/error.h>
#include <falcon/errhand.h>
#include <falcon/symbol.h>
#include <falcon/genericlist.h>
#include <fasm/pseudo.h>
#include <falcon/basealloc.h>
#include <falcon/stringstream.h>

#include <stdlib.h>

int fasm_parse( void *param );

namespace Falcon
{

class AsmLexer;
class  Module;

/** Falcon assembler.
   The assembler has the role to transform an assembly input in a sequence
   of bytecodes, while filling a module object with the module informations
   that are found on the way.

   The user of this class may then get the contents of the generated stream
   of bytes and add it to the module object (as its code), or to serialize
   the module and the code in sequence.
*/
class FALCON_DYN_CLASS AsmCompiler: public BaseAlloc
{
   AsmLexer *m_lexer;
   int m_errors;

   /** Map of label definitions.
      (const String *, LabelDef *)
   */
   Map m_labels;

   Symbol *m_current;

   Pseudo *m_switchItem;
   Pseudo *m_switchJump;

   Pseudo *m_switchEntryNil;
   Map m_switchEntriesInt;
   Map m_switchEntriesRng;
   Map m_switchEntriesStr;
   Map m_switchEntriesObj;
   List m_switchObjList;
   bool m_switchIsSelect;

   ErrorHandler *m_errhand;

   Module *m_module;
   StringStream *m_outTemp;
   uint32 m_currentLine;
   uint32 m_pc;

   void clearSwitch();
public:

   /** Builds the assembler object.

      The assembler requires a temporary growable and seekable support to
      store the code as it is being created. If the user of this class
      does not provide one invoking this function before starting the
      compilation, then a StringStream is used, and the whole
      compilation happens in memory. By providing a different seekable
      stream, i.e. a FileStream, the user is able to redirect the
      compilation to disk, sparing memory.

      This may be useful if memory is limited, and/or files to compile
      are big. In such cases, a compilation would require swap space,
      causing inefficient access to disk.

      Ift's responsibility of the caller to dispose of the streams
      after the compilation is complete.

      \param mod the module object.
      \param in the stream where the assembly is found.
   */
   AsmCompiler( Module *mod, Stream *in );

   virtual ~AsmCompiler();

   AsmLexer *lexer() const { return m_lexer; }

   void errorHandler( ErrorHandler *errhand ) { m_errhand = errhand; }
   ErrorHandler *errorHandler() const { return m_errhand; }

   bool compile();

   void addEntry();
   const String *addString( const String &data );
   void addGlobal( Pseudo *val, Pseudo *line, bool exp=false );
   void addVar( Pseudo *psym, Pseudo *pval, Pseudo *line, bool exp = false );
   void addConst( Pseudo *psym, Pseudo *pval, bool exp = false );
   void addAttrib( Pseudo *psym, Pseudo *line, bool exp = false );
   void addExport( Pseudo *val );
   void addLocal( Pseudo *val, Pseudo *line );
   void addFunction( Pseudo *val, Pseudo *line, bool exp=false );
   void addFuncDef( Pseudo *val, bool exp=false );
   void addParam( Pseudo *val, Pseudo *line );
   void addClass( Pseudo *val, Pseudo *line, bool exp=false );
   void addClassDef( Pseudo *val, bool exp = false );
   void addClassCtor( Pseudo *val );
   void addInstance( Pseudo *cls, Pseudo *object, Pseudo *line, bool exp = false );
   void addProperty( Pseudo *val, Pseudo *defval );
   void addPropRef( Pseudo *val, Pseudo *defval );
   void addFrom( Pseudo *val );
   LabelDef *addLabel( const String &name );
   void defineLabel( LabelDef *def );
   void addExtern( Pseudo *val, Pseudo *line );
   void addLoad( Pseudo *val, bool isFile = false );
   void addImport( Pseudo *val, Pseudo *line, Pseudo *mod=0, Pseudo *alias=0, bool isFile = false );
   void addDLine( Pseudo *line );

   void addInstr( unsigned char opCode, Pseudo *op1=0, Pseudo *op2=0, Pseudo *op3=0 );
   void setModuleName( Pseudo *val );
   void addFuncEnd();
   void addStatic();

   void addDSwitch( Pseudo *val, Pseudo *default_jump, bool bselect = false );
   void addDCase( Pseudo *val, Pseudo *jump, Pseudo *range_end=0 );
   void addDEndSwitch();
   void classHas( Pseudo *has );
   void classHasnt( Pseudo *hasnt );
   void addInherit( Pseudo *baseclass );

   void raiseError( int errorNum, int errorLine=0);
   void raiseError( int errorNum, const String &errorp, int errorLine=0);
   int errors() const { return m_errors; }
   void write_switch( Pseudo *op1, Pseudo *op2, Pseudo *oplist );
   bool defined( const String &sym ) const;
   Symbol *findSymbol( const String &name ) const;

   unsigned char paramDesc( Pseudo *op1 ) const;
   bool isParam( Pseudo *op1 ) const;
   bool isLocal( Pseudo *op1 ) const ;
   bool isExtern( Pseudo *op1 ) const ;
   void closeMain();

   Pseudo *regA_Inst();
   Pseudo *regB_Inst();
   Pseudo *regS1_Inst();
   Pseudo *regS2_Inst();
   Pseudo *regL1_Inst();
   Pseudo *regL2_Inst();
   Pseudo *nil_Inst();
};

void Pseudo_Deletor( void *pseudo_d );

} // end of namespace

#endif

/* end of comp.h */
