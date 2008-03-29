/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORMIDDLE = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SENDER = 281,
     SELF = 282,
     GIVE = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     LAMBDA = 292,
     INIT = 293,
     LOAD = 294,
     LAUNCH = 295,
     CONST_KW = 296,
     ATTRIBUTES = 297,
     PASS = 298,
     EXPORT = 299,
     IMPORT = 300,
     DIRECTIVE = 301,
     COLON = 302,
     FUNCDECL = 303,
     STATIC = 304,
     INNERFUNC = 305,
     FORDOT = 306,
     LISTPAR = 307,
     LOOP = 308,
     TRUE_TOKEN = 309,
     FALSE_TOKEN = 310,
     OUTER_STRING = 311,
     CLOSEPAR = 312,
     OPENPAR = 313,
     CLOSESQUARE = 314,
     OPENSQUARE = 315,
     DOT = 316,
     ARROW = 317,
     ASSIGN_POW = 318,
     ASSIGN_SHL = 319,
     ASSIGN_SHR = 320,
     ASSIGN_BXOR = 321,
     ASSIGN_BOR = 322,
     ASSIGN_BAND = 323,
     ASSIGN_MOD = 324,
     ASSIGN_DIV = 325,
     ASSIGN_MUL = 326,
     ASSIGN_SUB = 327,
     ASSIGN_ADD = 328,
     OP_EQ = 329,
     OP_TO = 330,
     COMMA = 331,
     QUESTION = 332,
     OR = 333,
     AND = 334,
     NOT = 335,
     LE = 336,
     GE = 337,
     LT = 338,
     GT = 339,
     NEQ = 340,
     EEQ = 341,
     PROVIDES = 342,
     OP_NOTIN = 343,
     OP_IN = 344,
     HASNT = 345,
     HAS = 346,
     DIESIS = 347,
     ATSIGN = 348,
     CAP = 349,
     VBAR = 350,
     AMPER = 351,
     MINUS = 352,
     PLUS = 353,
     PERCENT = 354,
     SLASH = 355,
     STAR = 356,
     POW = 357,
     SHR = 358,
     SHL = 359,
     BANG = 360,
     NEG = 361,
     DECREMENT = 362,
     INCREMENT = 363,
     DOLLAR = 364
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORMIDDLE 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SENDER 281
#define SELF 282
#define GIVE 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define LAMBDA 292
#define INIT 293
#define LOAD 294
#define LAUNCH 295
#define CONST_KW 296
#define ATTRIBUTES 297
#define PASS 298
#define EXPORT 299
#define IMPORT 300
#define DIRECTIVE 301
#define COLON 302
#define FUNCDECL 303
#define STATIC 304
#define INNERFUNC 305
#define FORDOT 306
#define LISTPAR 307
#define LOOP 308
#define TRUE_TOKEN 309
#define FALSE_TOKEN 310
#define OUTER_STRING 311
#define CLOSEPAR 312
#define OPENPAR 313
#define CLOSESQUARE 314
#define OPENSQUARE 315
#define DOT 316
#define ARROW 317
#define ASSIGN_POW 318
#define ASSIGN_SHL 319
#define ASSIGN_SHR 320
#define ASSIGN_BXOR 321
#define ASSIGN_BOR 322
#define ASSIGN_BAND 323
#define ASSIGN_MOD 324
#define ASSIGN_DIV 325
#define ASSIGN_MUL 326
#define ASSIGN_SUB 327
#define ASSIGN_ADD 328
#define OP_EQ 329
#define OP_TO 330
#define COMMA 331
#define QUESTION 332
#define OR 333
#define AND 334
#define NOT 335
#define LE 336
#define GE 337
#define LT 338
#define GT 339
#define NEQ 340
#define EEQ 341
#define PROVIDES 342
#define OP_NOTIN 343
#define OP_IN 344
#define HASNT 345
#define HAS 346
#define DIESIS 347
#define ATSIGN 348
#define CAP 349
#define VBAR 350
#define AMPER 351
#define MINUS 352
#define PLUS 353
#define PERCENT 354
#define SLASH 355
#define STAR 356
#define POW 357
#define SHR 358
#define SHL 359
#define BANG 360
#define NEG 361
#define DECREMENT 362
#define INCREMENT 363
#define DOLLAR 364




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 66 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
}
/* Line 1489 of yacc.c.  */
#line 281 "/home/gian/Progetti/falcon/core/engine/src_parser.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



