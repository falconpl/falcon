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
     FORDOT = 305,
     LISTPAR = 306,
     LOOP = 307,
     TRUE_TOKEN = 308,
     FALSE_TOKEN = 309,
     OUTER_STRING = 310,
     CLOSEPAR = 311,
     OPENPAR = 312,
     CLOSESQUARE = 313,
     OPENSQUARE = 314,
     DOT = 315,
     ASSIGN_POW = 316,
     ASSIGN_SHL = 317,
     ASSIGN_SHR = 318,
     ASSIGN_BXOR = 319,
     ASSIGN_BOR = 320,
     ASSIGN_BAND = 321,
     ASSIGN_MOD = 322,
     ASSIGN_DIV = 323,
     ASSIGN_MUL = 324,
     ASSIGN_SUB = 325,
     ASSIGN_ADD = 326,
     ARROW = 327,
     FOR_STEP = 328,
     OP_TO = 329,
     COMMA = 330,
     QUESTION = 331,
     OR = 332,
     AND = 333,
     NOT = 334,
     LET = 335,
     LE = 336,
     GE = 337,
     LT = 338,
     GT = 339,
     NEQ = 340,
     EEQ = 341,
     OP_EQ = 342,
     OP_ASSIGN = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     HASNT = 347,
     HAS = 348,
     DIESIS = 349,
     ATSIGN = 350,
     CAP = 351,
     VBAR = 352,
     AMPER = 353,
     MINUS = 354,
     PLUS = 355,
     PERCENT = 356,
     SLASH = 357,
     STAR = 358,
     POW = 359,
     SHR = 360,
     SHL = 361,
     BANG = 362,
     NEG = 363,
     DECREMENT = 364,
     INCREMENT = 365,
     DOLLAR = 366
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
#define FORDOT 305
#define LISTPAR 306
#define LOOP 307
#define TRUE_TOKEN 308
#define FALSE_TOKEN 309
#define OUTER_STRING 310
#define CLOSEPAR 311
#define OPENPAR 312
#define CLOSESQUARE 313
#define OPENSQUARE 314
#define DOT 315
#define ASSIGN_POW 316
#define ASSIGN_SHL 317
#define ASSIGN_SHR 318
#define ASSIGN_BXOR 319
#define ASSIGN_BOR 320
#define ASSIGN_BAND 321
#define ASSIGN_MOD 322
#define ASSIGN_DIV 323
#define ASSIGN_MUL 324
#define ASSIGN_SUB 325
#define ASSIGN_ADD 326
#define ARROW 327
#define FOR_STEP 328
#define OP_TO 329
#define COMMA 330
#define QUESTION 331
#define OR 332
#define AND 333
#define NOT 334
#define LET 335
#define LE 336
#define GE 337
#define LT 338
#define GT 339
#define NEQ 340
#define EEQ 341
#define OP_EQ 342
#define OP_ASSIGN 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define HASNT 347
#define HAS 348
#define DIESIS 349
#define ATSIGN 350
#define CAP 351
#define VBAR 352
#define AMPER 353
#define MINUS 354
#define PLUS 355
#define PERCENT 356
#define SLASH 357
#define STAR 358
#define POW 359
#define SHR 360
#define SHL 361
#define BANG 362
#define NEG 363
#define DECREMENT 364
#define INCREMENT 365
#define DOLLAR 366




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
#line 285 "/home/gian/Progetti/falcon/core/engine/src_parser.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



