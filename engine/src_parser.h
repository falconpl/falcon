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
     COLON = 300,
     FUNCDECL = 301,
     STATIC = 302,
     FORDOT = 303,
     LOOP = 304,
     OUTER_STRING = 305,
     CLOSEPAR = 306,
     OPENPAR = 307,
     CLOSESQUARE = 308,
     OPENSQUARE = 309,
     DOT = 310,
     ASSIGN_POW = 311,
     ASSIGN_SHL = 312,
     ASSIGN_SHR = 313,
     ASSIGN_BXOR = 314,
     ASSIGN_BOR = 315,
     ASSIGN_BAND = 316,
     ASSIGN_MOD = 317,
     ASSIGN_DIV = 318,
     ASSIGN_MUL = 319,
     ASSIGN_SUB = 320,
     ASSIGN_ADD = 321,
     ARROW = 322,
     FOR_STEP = 323,
     OP_TO = 324,
     COMMA = 325,
     QUESTION = 326,
     OR = 327,
     AND = 328,
     NOT = 329,
     LET = 330,
     LE = 331,
     GE = 332,
     LT = 333,
     GT = 334,
     NEQ = 335,
     EEQ = 336,
     OP_EQ = 337,
     OP_ASSIGN = 338,
     PROVIDES = 339,
     OP_NOTIN = 340,
     OP_IN = 341,
     HASNT = 342,
     HAS = 343,
     DIESIS = 344,
     ATSIGN = 345,
     CAP = 346,
     VBAR = 347,
     AMPER = 348,
     MINUS = 349,
     PLUS = 350,
     PERCENT = 351,
     SLASH = 352,
     STAR = 353,
     POW = 354,
     SHR = 355,
     SHL = 356,
     BANG = 357,
     NEG = 358,
     DECREMENT = 359,
     INCREMENT = 360,
     DOLLAR = 361
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
#define COLON 300
#define FUNCDECL 301
#define STATIC 302
#define FORDOT 303
#define LOOP 304
#define OUTER_STRING 305
#define CLOSEPAR 306
#define OPENPAR 307
#define CLOSESQUARE 308
#define OPENSQUARE 309
#define DOT 310
#define ASSIGN_POW 311
#define ASSIGN_SHL 312
#define ASSIGN_SHR 313
#define ASSIGN_BXOR 314
#define ASSIGN_BOR 315
#define ASSIGN_BAND 316
#define ASSIGN_MOD 317
#define ASSIGN_DIV 318
#define ASSIGN_MUL 319
#define ASSIGN_SUB 320
#define ASSIGN_ADD 321
#define ARROW 322
#define FOR_STEP 323
#define OP_TO 324
#define COMMA 325
#define QUESTION 326
#define OR 327
#define AND 328
#define NOT 329
#define LET 330
#define LE 331
#define GE 332
#define LT 333
#define GT 334
#define NEQ 335
#define EEQ 336
#define OP_EQ 337
#define OP_ASSIGN 338
#define PROVIDES 339
#define OP_NOTIN 340
#define OP_IN 341
#define HASNT 342
#define HAS 343
#define DIESIS 344
#define ATSIGN 345
#define CAP 346
#define VBAR 347
#define AMPER 348
#define MINUS 349
#define PLUS 350
#define PERCENT 351
#define SLASH 352
#define STAR 353
#define POW 354
#define SHR 355
#define SHL 356
#define BANG 357
#define NEG 358
#define DECREMENT 359
#define INCREMENT 360
#define DOLLAR 361




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 66 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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
#line 275 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



