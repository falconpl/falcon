/*
   FALCON - The Falcon Programming Language.
   FILE: faldisass.h

   Falcon disassembler private header file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun set 26 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon disassembler private header file.
   This is just an header file to keep falcon disassembler private types and
   data ordered.
*/

#ifndef flc_faldisass_H
#define flc_faldisass_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/common.h>
#include <falcon/pcodes.h>
#include <falcon/module.h>
#include <map>


namespace Falcon {

typedef enum {
   e_mode_comment,
   e_mode_table
} e_tabmode;


void write_operand( Stream *output, byte *instruction, int opnum, Module *mod );
uint32 calc_next( byte *instruction );
void disassembler( Module *module, Stream *out );
void write_strtable( e_tabmode mode , Stream *out, Module *mod );
void write_symtable( e_tabmode mode , Stream *out, const SymbolTable *st );
void write_deptable( e_tabmode mode , Stream *out, Module *mod );
void usage();

class Options
{
public:
   bool m_dump_str;
   bool m_dump_sym;
   bool m_dump_dep;
   bool m_isomorphic;
   bool m_inline_str;
   bool m_stdin;
   bool m_lineinfo;

   const char *m_fname;

   Options();
};

extern Options options;

typedef std::map< int32, int32 > t_labelMap;
typedef std::map< int32, const Symbol* > t_funcMap;
}

#endif

/* end of faldisass.h */
