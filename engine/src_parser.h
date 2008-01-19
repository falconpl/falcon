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
     FORALL = 276,
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
     DIRECTIVE = 300,
     COLON = 301,
     FUNCDECL = 302,
     STATIC = 303,
     FORDOT = 304,
     LISTPAR = 305,
     LOOP = 306,
     OUTER_STRING = 307,
     CLOSEPAR = 308,
     OPENPAR = 309,
     CLOSESQUARE = 310,
     OPENSQUARE = 311,
     DOT = 312,
     ASSIGN_POW = 313,
     ASSIGN_SHL = 314,
     ASSIGN_SHR = 315,
     ASSIGN_BXOR = 316,
     ASSIGN_BOR = 317,
     ASSIGN_BAND = 318,
     ASSIGN_MOD = 319,
     ASSIGN_DIV = 320,
     ASSIGN_MUL = 321,
     ASSIGN_SUB = 322,
     ASSIGN_ADD = 323,
     ARROW = 324,
     FOR_STEP = 325,
     OP_TO = 326,
     COMMA = 327,
     QUESTION = 328,
     OR = 329,
     AND = 330,
     NOT = 331,
     LET = 332,
     LE = 333,
     GE = 334,
     LT = 335,
     GT = 336,
     NEQ = 337,
     EEQ = 338,
     OP_EQ = 339,
     OP_ASSIGN = 340,
     PROVIDES = 341,
     OP_NOTIN = 342,
     OP_IN = 343,
     HASNT = 344,
     HAS = 345,
     DIESIS = 346,
     ATSIGN = 347,
     CAP = 348,
     VBAR = 349,
     AMPER = 350,
     MINUS = 351,
     PLUS = 352,
     PERCENT = 353,
     SLASH = 354,
     STAR = 355,
     POW = 356,
     SHR = 357,
     SHL = 358,
     BANG = 359,
     NEG = 360,
     DECREMENT = 361,
     INCREMENT = 362,
     DOLLAR = 363
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
#define FORALL 276
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
#define DIRECTIVE 300
#define COLON 301
#define FUNCDECL 302
#define STATIC 303
#define FORDOT 304
#define LISTPAR 305
#define LOOP 306
#define OUTER_STRING 307
#define CLOSEPAR 308
#define OPENPAR 309
#define CLOSESQUARE 310
#define OPENSQUARE 311
#define DOT 312
#define ASSIGN_POW 313
#define ASSIGN_SHL 314
#define ASSIGN_SHR 315
#define ASSIGN_BXOR 316
#define ASSIGN_BOR 317
#define ASSIGN_BAND 318
#define ASSIGN_MOD 319
#define ASSIGN_DIV 320
#define ASSIGN_MUL 321
#define ASSIGN_SUB 322
#define ASSIGN_ADD 323
#define ARROW 324
#define FOR_STEP 325
#define OP_TO 326
#define COMMA 327
#define QUESTION 328
#define OR 329
#define AND 330
#define NOT 331
#define LET 332
#define LE 333
#define GE 334
#define LT 335
#define GT 336
#define NEQ 337
#define EEQ 338
#define OP_EQ 339
#define OP_ASSIGN 340
#define PROVIDES 341
#define OP_NOTIN 342
#define OP_IN 343
#define HASNT 344
#define HAS 345
#define DIESIS 346
#define ATSIGN 347
#define CAP 348
#define VBAR 349
#define AMPER 350
#define MINUS 351
#define PLUS 352
#define PERCENT 353
#define SLASH 354
#define STAR 355
#define POW 356
#define SHR 357
#define SHL 358
#define BANG 359
#define NEG 360
#define DECREMENT 361
#define INCREMENT 362
#define DOLLAR 363




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 66 "/home/jeremy/Projects/falcon/core/engine/src_parser.yy"
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
#line 279 "/home/jeremy/Projects/falcon/core/engine/src_parser.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



