
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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
     UNB = 264,
     END = 265,
     DEF = 266,
     WHILE = 267,
     BREAK = 268,
     CONTINUE = 269,
     DROPPING = 270,
     IF = 271,
     ELSE = 272,
     ELIF = 273,
     FOR = 274,
     FORFIRST = 275,
     FORLAST = 276,
     FORMIDDLE = 277,
     SWITCH = 278,
     CASE = 279,
     DEFAULT = 280,
     SELECT = 281,
     SELF = 282,
     FSELF = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     INIT = 292,
     LOAD = 293,
     LAUNCH = 294,
     CONST_KW = 295,
     EXPORT = 296,
     IMPORT = 297,
     DIRECTIVE = 298,
     COLON = 299,
     FUNCDECL = 300,
     STATIC = 301,
     INNERFUNC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     ENUM = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_AS = 331,
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     OP_EXEQ = 344,
     PROVIDES = 345,
     OP_NOTIN = 346,
     OP_IN = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     CAP_XOROOB = 361,
     CAP_ISOOB = 362,
     CAP_DEOOB = 363,
     CAP_OOB = 364,
     CAP_EVAL = 365,
     TILDE = 366,
     NEG = 367,
     AMPER = 368,
     DECREMENT = 369,
     INCREMENT = 370,
     DOLLAR = 371
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define UNB 264
#define END 265
#define DEF 266
#define WHILE 267
#define BREAK 268
#define CONTINUE 269
#define DROPPING 270
#define IF 271
#define ELSE 272
#define ELIF 273
#define FOR 274
#define FORFIRST 275
#define FORLAST 276
#define FORMIDDLE 277
#define SWITCH 278
#define CASE 279
#define DEFAULT 280
#define SELECT 281
#define SELF 282
#define FSELF 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define INIT 292
#define LOAD 293
#define LAUNCH 294
#define CONST_KW 295
#define EXPORT 296
#define IMPORT 297
#define DIRECTIVE 298
#define COLON 299
#define FUNCDECL 300
#define STATIC 301
#define INNERFUNC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define ENUM 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_AS 331
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define OP_EXEQ 344
#define PROVIDES 345
#define OP_NOTIN 346
#define OP_IN 347
#define DIESIS 348
#define ATSIGN 349
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define CAP_XOROOB 361
#define CAP_ISOOB 362
#define CAP_DEOOB 363
#define CAP_OOB 364
#define CAP_EVAL 365
#define TILDE 366
#define NEG 367
#define AMPER 368
#define DECREMENT 369
#define INCREMENT 370
#define DOLLAR 371




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union 
/* Line 1676 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t
{

/* Line 1676 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"

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
   Falcon::List *fal_genericList;



/* Line 1676 of yacc.c  */
#line 304 "/home/gian/Progetti/falcon/core/engine/src_parser.hpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif




