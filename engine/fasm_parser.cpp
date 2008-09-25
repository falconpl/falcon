/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse fasm_parse
#define yylex   fasm_lex
#define yyerror fasm_error
#define yylval  fasm_lval
#define yychar  fasm_char
#define yydebug fasm_debug
#define yynerrs fasm_nerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     NAME = 259,
     COMMA = 260,
     COLON = 261,
     DENTRY = 262,
     DGLOBAL = 263,
     DVAR = 264,
     DCONST = 265,
     DATTRIB = 266,
     DLOCAL = 267,
     DPARAM = 268,
     DFUNCDEF = 269,
     DENDFUNC = 270,
     DFUNC = 271,
     DMETHOD = 272,
     DPROP = 273,
     DPROPREF = 274,
     DCLASS = 275,
     DCLASSDEF = 276,
     DCTOR = 277,
     DENDCLASS = 278,
     DFROM = 279,
     DEXTERN = 280,
     DMODULE = 281,
     DLOAD = 282,
     DIMPORT = 283,
     DSWITCH = 284,
     DSELECT = 285,
     DCASE = 286,
     DENDSWITCH = 287,
     DLINE = 288,
     DSTRING = 289,
     DISTRING = 290,
     DCSTRING = 291,
     DHAS = 292,
     DHASNT = 293,
     DINHERIT = 294,
     DINSTANCE = 295,
     SYMBOL = 296,
     EXPORT = 297,
     LABEL = 298,
     INTEGER = 299,
     REG_A = 300,
     REG_B = 301,
     REG_S1 = 302,
     REG_S2 = 303,
     REG_L1 = 304,
     REG_L2 = 305,
     NUMERIC = 306,
     STRING = 307,
     STRING_ID = 308,
     TRUE_TOKEN = 309,
     FALSE_TOKEN = 310,
     I_LD = 311,
     I_LNIL = 312,
     NIL = 313,
     I_ADD = 314,
     I_SUB = 315,
     I_MUL = 316,
     I_DIV = 317,
     I_MOD = 318,
     I_POW = 319,
     I_ADDS = 320,
     I_SUBS = 321,
     I_MULS = 322,
     I_DIVS = 323,
     I_POWS = 324,
     I_INC = 325,
     I_DEC = 326,
     I_INCP = 327,
     I_DECP = 328,
     I_NEG = 329,
     I_NOT = 330,
     I_RET = 331,
     I_RETV = 332,
     I_RETA = 333,
     I_FORK = 334,
     I_PUSH = 335,
     I_PSHR = 336,
     I_PSHN = 337,
     I_POP = 338,
     I_LDV = 339,
     I_LDVT = 340,
     I_STV = 341,
     I_STVR = 342,
     I_STVS = 343,
     I_LDP = 344,
     I_LDPT = 345,
     I_STP = 346,
     I_STPR = 347,
     I_STPS = 348,
     I_TRAV = 349,
     I_TRAN = 350,
     I_TRAL = 351,
     I_IPOP = 352,
     I_XPOP = 353,
     I_GENA = 354,
     I_GEND = 355,
     I_GENR = 356,
     I_GEOR = 357,
     I_JMP = 358,
     I_IFT = 359,
     I_IFF = 360,
     I_BOOL = 361,
     I_EQ = 362,
     I_NEQ = 363,
     I_GT = 364,
     I_GE = 365,
     I_LT = 366,
     I_LE = 367,
     I_UNPK = 368,
     I_UNPS = 369,
     I_CALL = 370,
     I_INST = 371,
     I_SWCH = 372,
     I_SELE = 373,
     I_NOP = 374,
     I_TRY = 375,
     I_JTRY = 376,
     I_PTRY = 377,
     I_RIS = 378,
     I_LDRF = 379,
     I_ONCE = 380,
     I_BAND = 381,
     I_BOR = 382,
     I_BXOR = 383,
     I_BNOT = 384,
     I_MODS = 385,
     I_AND = 386,
     I_OR = 387,
     I_ANDS = 388,
     I_ORS = 389,
     I_XORS = 390,
     I_NOTS = 391,
     I_HAS = 392,
     I_HASN = 393,
     I_GIVE = 394,
     I_GIVN = 395,
     I_IN = 396,
     I_NOIN = 397,
     I_PROV = 398,
     I_END = 399,
     I_PEEK = 400,
     I_PSIN = 401,
     I_PASS = 402,
     I_SHR = 403,
     I_SHL = 404,
     I_SHRS = 405,
     I_SHLS = 406,
     I_LDVR = 407,
     I_LDPR = 408,
     I_LSB = 409,
     I_INDI = 410,
     I_STEX = 411,
     I_TRAC = 412,
     I_WRT = 413,
     I_STO = 414,
     I_FORB = 415
   };
#endif
/* Tokens.  */
#define EOL 258
#define NAME 259
#define COMMA 260
#define COLON 261
#define DENTRY 262
#define DGLOBAL 263
#define DVAR 264
#define DCONST 265
#define DATTRIB 266
#define DLOCAL 267
#define DPARAM 268
#define DFUNCDEF 269
#define DENDFUNC 270
#define DFUNC 271
#define DMETHOD 272
#define DPROP 273
#define DPROPREF 274
#define DCLASS 275
#define DCLASSDEF 276
#define DCTOR 277
#define DENDCLASS 278
#define DFROM 279
#define DEXTERN 280
#define DMODULE 281
#define DLOAD 282
#define DIMPORT 283
#define DSWITCH 284
#define DSELECT 285
#define DCASE 286
#define DENDSWITCH 287
#define DLINE 288
#define DSTRING 289
#define DISTRING 290
#define DCSTRING 291
#define DHAS 292
#define DHASNT 293
#define DINHERIT 294
#define DINSTANCE 295
#define SYMBOL 296
#define EXPORT 297
#define LABEL 298
#define INTEGER 299
#define REG_A 300
#define REG_B 301
#define REG_S1 302
#define REG_S2 303
#define REG_L1 304
#define REG_L2 305
#define NUMERIC 306
#define STRING 307
#define STRING_ID 308
#define TRUE_TOKEN 309
#define FALSE_TOKEN 310
#define I_LD 311
#define I_LNIL 312
#define NIL 313
#define I_ADD 314
#define I_SUB 315
#define I_MUL 316
#define I_DIV 317
#define I_MOD 318
#define I_POW 319
#define I_ADDS 320
#define I_SUBS 321
#define I_MULS 322
#define I_DIVS 323
#define I_POWS 324
#define I_INC 325
#define I_DEC 326
#define I_INCP 327
#define I_DECP 328
#define I_NEG 329
#define I_NOT 330
#define I_RET 331
#define I_RETV 332
#define I_RETA 333
#define I_FORK 334
#define I_PUSH 335
#define I_PSHR 336
#define I_PSHN 337
#define I_POP 338
#define I_LDV 339
#define I_LDVT 340
#define I_STV 341
#define I_STVR 342
#define I_STVS 343
#define I_LDP 344
#define I_LDPT 345
#define I_STP 346
#define I_STPR 347
#define I_STPS 348
#define I_TRAV 349
#define I_TRAN 350
#define I_TRAL 351
#define I_IPOP 352
#define I_XPOP 353
#define I_GENA 354
#define I_GEND 355
#define I_GENR 356
#define I_GEOR 357
#define I_JMP 358
#define I_IFT 359
#define I_IFF 360
#define I_BOOL 361
#define I_EQ 362
#define I_NEQ 363
#define I_GT 364
#define I_GE 365
#define I_LT 366
#define I_LE 367
#define I_UNPK 368
#define I_UNPS 369
#define I_CALL 370
#define I_INST 371
#define I_SWCH 372
#define I_SELE 373
#define I_NOP 374
#define I_TRY 375
#define I_JTRY 376
#define I_PTRY 377
#define I_RIS 378
#define I_LDRF 379
#define I_ONCE 380
#define I_BAND 381
#define I_BOR 382
#define I_BXOR 383
#define I_BNOT 384
#define I_MODS 385
#define I_AND 386
#define I_OR 387
#define I_ANDS 388
#define I_ORS 389
#define I_XORS 390
#define I_NOTS 391
#define I_HAS 392
#define I_HASN 393
#define I_GIVE 394
#define I_GIVN 395
#define I_IN 396
#define I_NOIN 397
#define I_PROV 398
#define I_END 399
#define I_PEEK 400
#define I_PSIN 401
#define I_PASS 402
#define I_SHR 403
#define I_SHL 404
#define I_SHRS 405
#define I_SHLS 406
#define I_LDVR 407
#define I_LDPR 408
#define I_LSB 409
#define I_INDI 410
#define I_STEX 411
#define I_TRAC 412
#define I_WRT 413
#define I_STO 414
#define I_FORB 415




/* Copy the first part of user declarations.  */
#line 17 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"

#include <falcon/setup.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>

#include <fasm/comp.h>
#include <fasm/clexer.h>
#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/pcodes.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::AsmCompiler *>(fasm_param) )
#define  LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM fasm_param
#define YYLEX_PARAM fasm_param
#define YYSTYPE Falcon::Pseudo *

int fasm_parse( void *param );
void fasm_error( const char *s );

inline int yylex (void *lvalp, void *fasm_param)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 472 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1836

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  161
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  125
/* YYNRULES -- Number of rules.  */
#define YYNRULES  452
/* YYNRULES -- Number of states.  */
#define YYNSTATES  814

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   415

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    67,    69,    71,    73,    75,    77,    80,    84,    89,
      94,   100,   104,   109,   113,   118,   122,   126,   129,   133,
     137,   142,   144,   147,   150,   154,   159,   164,   170,   176,
     181,   186,   191,   196,   201,   206,   211,   216,   221,   228,
     230,   234,   238,   242,   245,   248,   253,   259,   263,   268,
     271,   275,   278,   280,   281,   286,   289,   293,   296,   299,
     302,   305,   307,   311,   313,   317,   320,   321,   323,   327,
     329,   331,   332,   334,   336,   338,   340,   342,   344,   346,
     348,   350,   352,   354,   356,   358,   360,   362,   364,   366,
     368,   370,   372,   374,   376,   378,   380,   382,   384,   386,
     388,   390,   392,   394,   396,   398,   400,   402,   404,   406,
     408,   410,   412,   414,   416,   418,   420,   422,   424,   426,
     428,   430,   432,   434,   436,   438,   440,   442,   444,   446,
     448,   450,   452,   454,   456,   458,   460,   462,   464,   466,
     468,   470,   472,   474,   476,   478,   480,   482,   484,   486,
     488,   490,   492,   494,   496,   498,   500,   502,   504,   506,
     508,   510,   512,   514,   516,   518,   520,   522,   524,   526,
     528,   530,   532,   534,   536,   538,   540,   545,   548,   553,
     556,   559,   562,   567,   570,   575,   578,   583,   586,   591,
     594,   599,   602,   607,   610,   615,   618,   623,   626,   631,
     634,   639,   642,   647,   650,   655,   658,   663,   666,   671,
     674,   679,   682,   687,   690,   693,   696,   699,   702,   705,
     708,   711,   714,   717,   720,   723,   726,   729,   732,   735,
     740,   745,   748,   753,   758,   761,   766,   769,   774,   777,
     780,   782,   785,   788,   791,   794,   797,   800,   803,   806,
     809,   814,   819,   822,   829,   836,   839,   846,   853,   856,
     863,   870,   873,   878,   883,   886,   891,   896,   899,   906,
     913,   916,   923,   930,   933,   940,   947,   950,   955,   960,
     963,   970,   977,   980,   987,   994,   997,  1000,  1003,  1006,
    1009,  1012,  1015,  1018,  1021,  1024,  1031,  1034,  1037,  1040,
    1043,  1046,  1049,  1052,  1055,  1058,  1061,  1066,  1071,  1074,
    1079,  1084,  1087,  1092,  1097,  1100,  1103,  1106,  1109,  1111,
    1114,  1116,  1119,  1122,  1125,  1127,  1130,  1133,  1136,  1138,
    1141,  1148,  1151,  1158,  1161,  1165,  1171,  1176,  1181,  1184,
    1189,  1194,  1199,  1204,  1207,  1212,  1217,  1222,  1227,  1230,
    1235,  1240,  1245,  1250,  1253,  1256,  1259,  1262,  1267,  1270,
    1275,  1278,  1283,  1286,  1291,  1294,  1299,  1302,  1307,  1310,
    1315,  1318,  1321,  1324,  1329,  1334,  1337,  1342,  1347,  1350,
    1355,  1360,  1363,  1368,  1373,  1376,  1381,  1384,  1389,  1392,
    1397,  1400,  1403,  1406,  1409,  1412,  1417,  1420,  1425,  1428,
    1433,  1436,  1441,  1444,  1449,  1452,  1457,  1460,  1465,  1468,
    1471,  1474,  1477,  1480,  1483,  1486,  1489,  1492,  1495,  1498,
    1503,  1506,  1511
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     162,     0,    -1,    -1,   162,   163,    -1,     3,    -1,   173,
       3,    -1,   177,   181,     3,    -1,   177,     3,    -1,   181,
       3,    -1,     1,     3,    -1,    58,    -1,   165,    -1,   166,
      -1,   169,    -1,    41,    -1,   167,    -1,    45,    -1,    46,
      -1,    47,    -1,    48,    -1,    49,    -1,    50,    -1,    58,
      -1,   169,    -1,    51,    -1,    54,    -1,    55,    -1,   170,
      -1,    53,    -1,    52,    -1,    44,    -1,    53,    -1,    52,
      -1,    52,    -1,     4,    -1,     7,    -1,    26,     4,    -1,
       8,     4,   180,    -1,     8,     4,   180,    42,    -1,     9,
       4,   168,   180,    -1,     9,     4,   168,   180,    42,    -1,
      10,     4,   168,    -1,    10,     4,   168,    42,    -1,    11,
       4,   180,    -1,    11,     4,   180,    42,    -1,    12,     4,
     180,    -1,    13,     4,   180,    -1,    14,     4,    -1,    14,
       4,    42,    -1,    16,     4,   180,    -1,    16,     4,   180,
      42,    -1,    15,    -1,    27,     4,    -1,    27,    52,    -1,
      28,     4,   180,    -1,    28,     4,   180,     4,    -1,    28,
       4,   180,    52,    -1,    28,     4,   180,     4,   172,    -1,
      28,     4,   180,    52,   172,    -1,    29,   166,     5,     4,
      -1,    29,   166,     5,    44,    -1,    30,   166,     5,     4,
      -1,    30,   166,     5,    44,    -1,    31,    58,     5,     4,
      -1,    31,    44,     5,     4,    -1,    31,    52,     5,     4,
      -1,    31,    53,     5,     4,    -1,    31,    41,     5,     4,
      -1,    31,    44,     6,    44,     5,     4,    -1,    32,    -1,
      18,     4,   168,    -1,    18,     4,    41,    -1,    19,     4,
      41,    -1,    37,   175,    -1,    38,   176,    -1,    40,    41,
       4,   180,    -1,    40,    41,     4,   180,    42,    -1,    20,
       4,   180,    -1,    20,     4,   180,    42,    -1,    21,     4,
      -1,    21,     4,    42,    -1,    22,    41,    -1,    23,    -1,
      -1,    39,    41,   174,   178,    -1,    24,     4,    -1,    25,
       4,   180,    -1,    33,    44,    -1,    34,    52,    -1,    35,
      52,    -1,    36,    52,    -1,    41,    -1,   175,     5,    41,
      -1,    41,    -1,   176,     5,    41,    -1,    43,     6,    -1,
      -1,   179,    -1,   178,     5,   179,    -1,   168,    -1,    41,
      -1,    -1,    44,    -1,   182,    -1,   184,    -1,   185,    -1,
     187,    -1,   189,    -1,   191,    -1,   193,    -1,   194,    -1,
     186,    -1,   188,    -1,   190,    -1,   192,    -1,   202,    -1,
     203,    -1,   204,    -1,   205,    -1,   195,    -1,   196,    -1,
     197,    -1,   198,    -1,   199,    -1,   200,    -1,   206,    -1,
     207,    -1,   212,    -1,   213,    -1,   214,    -1,   216,    -1,
     217,    -1,   218,    -1,   219,    -1,   220,    -1,   221,    -1,
     222,    -1,   223,    -1,   224,    -1,   226,    -1,   225,    -1,
     227,    -1,   228,    -1,   229,    -1,   230,    -1,   231,    -1,
     232,    -1,   233,    -1,   234,    -1,   236,    -1,   237,    -1,
     238,    -1,   239,    -1,   240,    -1,   210,    -1,   211,    -1,
     208,    -1,   209,    -1,   242,    -1,   244,    -1,   243,    -1,
     245,    -1,   201,    -1,   246,    -1,   241,    -1,   235,    -1,
     248,    -1,   249,    -1,   183,    -1,   247,    -1,   252,    -1,
     253,    -1,   254,    -1,   255,    -1,   256,    -1,   257,    -1,
     258,    -1,   259,    -1,   260,    -1,   263,    -1,   261,    -1,
     262,    -1,   264,    -1,   265,    -1,   266,    -1,   267,    -1,
     268,    -1,   269,    -1,   270,    -1,   251,    -1,   215,    -1,
     271,    -1,   272,    -1,   273,    -1,   274,    -1,   275,    -1,
     276,    -1,   277,    -1,   278,    -1,   279,    -1,   280,    -1,
     281,    -1,   282,    -1,   283,    -1,   284,    -1,   285,    -1,
      56,   166,     5,   164,    -1,    56,     1,    -1,   124,   166,
       5,   164,    -1,   124,     1,    -1,    57,   166,    -1,    57,
       1,    -1,    59,   164,     5,   164,    -1,    59,     1,    -1,
      65,   166,     5,   164,    -1,    65,     1,    -1,    60,   164,
       5,   164,    -1,    60,     1,    -1,    66,   166,     5,   164,
      -1,    66,     1,    -1,    61,   164,     5,   164,    -1,    61,
       1,    -1,    67,   166,     5,   164,    -1,    67,     1,    -1,
      62,   164,     5,   164,    -1,    62,     1,    -1,    68,   166,
       5,   164,    -1,    68,     1,    -1,    63,   164,     5,   164,
      -1,    63,     1,    -1,    64,   164,     5,   164,    -1,    64,
       1,    -1,   107,   164,     5,   164,    -1,   107,     1,    -1,
     108,   164,     5,   164,    -1,   108,     1,    -1,   110,   164,
       5,   164,    -1,   110,     1,    -1,   109,   164,     5,   164,
      -1,   109,     1,    -1,   112,   164,     5,   164,    -1,   112,
       1,    -1,   111,   164,     5,   164,    -1,   111,     1,    -1,
     120,     4,    -1,   120,    44,    -1,   120,     1,    -1,    70,
     166,    -1,    70,     1,    -1,    71,   166,    -1,    71,     1,
      -1,    72,   166,    -1,    72,     1,    -1,    73,   166,    -1,
      73,     1,    -1,    74,   164,    -1,    74,     1,    -1,    75,
     164,    -1,    75,     1,    -1,   115,    44,     5,   166,    -1,
     115,    44,     5,     4,    -1,   115,     1,    -1,   116,    44,
       5,   166,    -1,   116,    44,     5,     4,    -1,   116,     1,
      -1,   113,   166,     5,   166,    -1,   113,     1,    -1,   114,
      44,     5,   166,    -1,   114,     1,    -1,    80,   164,    -1,
      82,    -1,    80,     1,    -1,    81,   164,    -1,    81,     1,
      -1,    83,   166,    -1,    83,     1,    -1,   145,   166,    -1,
     145,     1,    -1,    98,   166,    -1,    98,     1,    -1,    84,
     166,     5,   166,    -1,    84,   166,     5,   170,    -1,    84,
       1,    -1,    85,   166,     5,   166,     5,   166,    -1,    85,
     166,     5,   170,     5,   166,    -1,    85,     1,    -1,    86,
     166,     5,   166,     5,   164,    -1,    86,   166,     5,   170,
       5,   164,    -1,    86,     1,    -1,    87,   166,     5,   166,
       5,   164,    -1,    87,   166,     5,   170,     5,   164,    -1,
      87,     1,    -1,    88,   166,     5,   166,    -1,    88,   166,
       5,   170,    -1,    88,     1,    -1,    89,   166,     5,   166,
      -1,    89,   166,     5,   170,    -1,    89,     1,    -1,    90,
     166,     5,   166,     5,   166,    -1,    90,   166,     5,   170,
       5,   166,    -1,    90,     1,    -1,    91,   166,     5,   166,
       5,   164,    -1,    91,   166,     5,   170,     5,   164,    -1,
      91,     1,    -1,    92,   166,     5,   166,     5,   166,    -1,
      92,   166,     5,   170,     5,   166,    -1,    92,     1,    -1,
      93,   166,     5,   166,    -1,    93,   166,     5,   170,    -1,
      93,     1,    -1,    94,    44,     5,   164,     5,   164,    -1,
      94,     4,     5,   164,     5,   164,    -1,    94,     1,    -1,
      95,     4,     5,     4,     5,    44,    -1,    95,    44,     5,
      44,     5,    44,    -1,    95,     1,    -1,    96,    44,    -1,
      96,     4,    -1,    96,     1,    -1,    97,    44,    -1,    97,
       1,    -1,    99,    44,    -1,    99,     1,    -1,   100,    44,
      -1,   100,     1,    -1,   101,   165,     5,   165,     5,   168,
      -1,   101,     1,    -1,   102,   165,    -1,   102,     1,    -1,
     123,   164,    -1,   123,     1,    -1,   103,     4,    -1,   103,
      44,    -1,   103,     1,    -1,   106,   164,    -1,   106,     1,
      -1,   104,     4,     5,   164,    -1,   104,    44,     5,   164,
      -1,   104,     1,    -1,   105,     4,     5,   164,    -1,   105,
      44,     5,   164,    -1,   105,     1,    -1,    79,    44,     5,
       4,    -1,    79,    44,     5,    44,    -1,    79,     1,    -1,
     121,     4,    -1,   121,    44,    -1,   121,     1,    -1,    76,
      -1,    76,     1,    -1,    78,    -1,    78,     1,    -1,    77,
     164,    -1,    77,     1,    -1,   119,    -1,   119,     1,    -1,
     122,    44,    -1,   122,     1,    -1,   144,    -1,   144,     1,
      -1,   117,    44,     5,   166,     5,   250,    -1,   117,     1,
      -1,   118,    44,     5,   166,     5,   250,    -1,   118,     1,
      -1,    44,     5,     4,    -1,   250,     5,    44,     5,     4,
      -1,   125,    44,     5,   164,    -1,   125,     4,     5,   164,
      -1,   125,     1,    -1,   126,    41,     5,    41,    -1,   126,
      41,     5,    44,    -1,   126,    44,     5,    41,    -1,   126,
      44,     5,    44,    -1,   126,     1,    -1,   127,    41,     5,
      41,    -1,   127,    41,     5,    44,    -1,   127,    44,     5,
      41,    -1,   127,    44,     5,    44,    -1,   127,     1,    -1,
     128,    41,     5,    41,    -1,   128,    41,     5,    44,    -1,
     128,    44,     5,    41,    -1,   128,    44,     5,    44,    -1,
     128,     1,    -1,   129,    41,    -1,   129,    44,    -1,   129,
       1,    -1,   131,   164,     5,   164,    -1,   131,     1,    -1,
     132,   164,     5,   164,    -1,   132,     1,    -1,   133,   166,
       5,   165,    -1,   133,     1,    -1,   134,   166,     5,   165,
      -1,   134,     1,    -1,   135,   166,     5,   165,    -1,   135,
       1,    -1,   130,   166,     5,   165,    -1,   130,     1,    -1,
      69,   166,     5,   165,    -1,    69,     1,    -1,   136,   166,
      -1,   136,     1,    -1,   137,   166,     5,   166,    -1,   137,
     166,     5,    44,    -1,   137,     1,    -1,   138,   166,     5,
     166,    -1,   138,   166,     5,    44,    -1,   138,     1,    -1,
     139,   166,     5,   166,    -1,   139,   166,     5,    44,    -1,
     139,     1,    -1,   140,   166,     5,   166,    -1,   140,   166,
       5,    44,    -1,   140,     1,    -1,   141,   164,     5,   165,
      -1,   141,     1,    -1,   142,   164,     5,   165,    -1,   142,
       1,    -1,   143,   164,     5,   165,    -1,   143,     1,    -1,
     146,   166,    -1,   146,     1,    -1,   147,   166,    -1,   147,
       1,    -1,   148,   164,     5,   164,    -1,   148,     1,    -1,
     149,   164,     5,   164,    -1,   149,     1,    -1,   150,   164,
       5,   164,    -1,   150,     1,    -1,   151,   164,     5,   164,
      -1,   151,     1,    -1,   152,   164,     5,   164,    -1,   152,
       1,    -1,   153,   164,     5,   164,    -1,   153,     1,    -1,
     154,   164,     5,   164,    -1,   154,     1,    -1,   155,   171,
      -1,   155,   166,    -1,   155,     1,    -1,   156,   171,    -1,
     156,   166,    -1,   156,     1,    -1,   157,   164,    -1,   157,
       1,    -1,   158,   164,    -1,   158,     1,    -1,   159,   166,
       5,   164,    -1,   159,     1,    -1,   160,   171,     5,   164,
      -1,   160,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   231,   231,   233,   237,   238,   239,   240,   241,   242,
     245,   245,   246,   246,   247,   247,   248,   248,   248,   248,
     248,   248,   250,   250,   251,   251,   251,   251,   252,   252,
     252,   253,   253,   254,   254,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   283,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   310,   311,   312,   313,   314,   319,
     325,   334,   335,   339,   340,   343,   345,   347,   348,   352,
     353,   357,   358,   362,   363,   364,   365,   366,   367,   368,
     369,   370,   371,   372,   373,   374,   375,   376,   377,   378,
     379,   380,   381,   382,   383,   384,   385,   386,   387,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,   406,   407,   408,
     409,   410,   411,   412,   413,   414,   415,   416,   417,   418,
     419,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,   445,   446,   447,   448,
     449,   450,   451,   452,   453,   454,   455,   456,   457,   458,
     459,   460,   461,   462,   463,   464,   468,   469,   473,   474,
     478,   479,   483,   484,   488,   489,   494,   495,   499,   500,
     504,   505,   509,   510,   515,   516,   520,   521,   525,   526,
     530,   531,   536,   537,   541,   542,   546,   547,   551,   552,
     556,   557,   561,   562,   566,   567,   568,   572,   573,   577,
     578,   583,   584,   588,   589,   594,   595,   599,   600,   604,
     605,   606,   610,   611,   612,   616,   617,   621,   622,   627,
     628,   629,   633,   634,   639,   640,   644,   645,   649,   650,
     655,   656,   657,   661,   662,   663,   667,   668,   669,   673,
     674,   675,   679,   680,   681,   685,   686,   687,   691,   692,
     693,   697,   698,   699,   703,   704,   705,   709,   710,   711,
     715,   716,   717,   721,   722,   723,   727,   728,   729,   733,
     734,   738,   739,   743,   744,   748,   749,   753,   754,   758,
     759,   763,   764,   765,   769,   770,   774,   775,   776,   780,
     781,   782,   787,   788,   789,   793,   794,   795,   799,   800,
     804,   805,   809,   810,   814,   815,   819,   820,   824,   825,
     829,   830,   834,   835,   839,   848,   857,   858,   859,   863,
     864,   865,   866,   867,   871,   872,   873,   874,   875,   879,
     880,   881,   882,   883,   887,   888,   889,   893,   894,   898,
     899,   903,   904,   908,   909,   913,   914,   918,   919,   923,
     924,   928,   929,   933,   934,   935,   939,   940,   941,   945,
     946,   947,   951,   952,   953,   958,   959,   963,   964,   968,
     969,   973,   974,   978,   979,   983,   984,   988,   989,   993,
     994,   998,   999,  1003,  1004,  1008,  1009,  1013,  1014,  1018,
    1019,  1020,  1024,  1025,  1026,  1030,  1031,  1035,  1036,  1041,
    1042,  1046,  1047
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "NAME", "COMMA", "COLON",
  "DENTRY", "DGLOBAL", "DVAR", "DCONST", "DATTRIB", "DLOCAL", "DPARAM",
  "DFUNCDEF", "DENDFUNC", "DFUNC", "DMETHOD", "DPROP", "DPROPREF",
  "DCLASS", "DCLASSDEF", "DCTOR", "DENDCLASS", "DFROM", "DEXTERN",
  "DMODULE", "DLOAD", "DIMPORT", "DSWITCH", "DSELECT", "DCASE",
  "DENDSWITCH", "DLINE", "DSTRING", "DISTRING", "DCSTRING", "DHAS",
  "DHASNT", "DINHERIT", "DINSTANCE", "SYMBOL", "EXPORT", "LABEL",
  "INTEGER", "REG_A", "REG_B", "REG_S1", "REG_S2", "REG_L1", "REG_L2",
  "NUMERIC", "STRING", "STRING_ID", "TRUE_TOKEN", "FALSE_TOKEN", "I_LD",
  "I_LNIL", "NIL", "I_ADD", "I_SUB", "I_MUL", "I_DIV", "I_MOD", "I_POW",
  "I_ADDS", "I_SUBS", "I_MULS", "I_DIVS", "I_POWS", "I_INC", "I_DEC",
  "I_INCP", "I_DECP", "I_NEG", "I_NOT", "I_RET", "I_RETV", "I_RETA",
  "I_FORK", "I_PUSH", "I_PSHR", "I_PSHN", "I_POP", "I_LDV", "I_LDVT",
  "I_STV", "I_STVR", "I_STVS", "I_LDP", "I_LDPT", "I_STP", "I_STPR",
  "I_STPS", "I_TRAV", "I_TRAN", "I_TRAL", "I_IPOP", "I_XPOP", "I_GENA",
  "I_GEND", "I_GENR", "I_GEOR", "I_JMP", "I_IFT", "I_IFF", "I_BOOL",
  "I_EQ", "I_NEQ", "I_GT", "I_GE", "I_LT", "I_LE", "I_UNPK", "I_UNPS",
  "I_CALL", "I_INST", "I_SWCH", "I_SELE", "I_NOP", "I_TRY", "I_JTRY",
  "I_PTRY", "I_RIS", "I_LDRF", "I_ONCE", "I_BAND", "I_BOR", "I_BXOR",
  "I_BNOT", "I_MODS", "I_AND", "I_OR", "I_ANDS", "I_ORS", "I_XORS",
  "I_NOTS", "I_HAS", "I_HASN", "I_GIVE", "I_GIVN", "I_IN", "I_NOIN",
  "I_PROV", "I_END", "I_PEEK", "I_PSIN", "I_PASS", "I_SHR", "I_SHL",
  "I_SHRS", "I_SHLS", "I_LDVR", "I_LDPR", "I_LSB", "I_INDI", "I_STEX",
  "I_TRAC", "I_WRT", "I_STO", "I_FORB", "$accept", "input", "line",
  "xoperand", "operand", "op_variable", "op_register", "x_op_immediate",
  "op_immediate", "op_scalar", "op_string", "string_or_name", "directive",
  "@1", "has_symlist", "hasnt_symlist", "label", "inherit_param_list",
  "inherit_param", "def_line", "instruction", "inst_ld", "inst_ldrf",
  "inst_ldnil", "inst_add", "inst_adds", "inst_sub", "inst_subs",
  "inst_mul", "inst_muls", "inst_div", "inst_divs", "inst_mod", "inst_pow",
  "inst_eq", "inst_ne", "inst_ge", "inst_gt", "inst_le", "inst_lt",
  "inst_try", "inst_inc", "inst_dec", "inst_incp", "inst_decp", "inst_neg",
  "inst_not", "inst_call", "inst_inst", "inst_unpk", "inst_unps",
  "inst_push", "inst_pshr", "inst_pop", "inst_peek", "inst_xpop",
  "inst_ldv", "inst_ldvt", "inst_stv", "inst_stvr", "inst_stvs",
  "inst_ldp", "inst_ldpt", "inst_stp", "inst_stpr", "inst_stps",
  "inst_trav", "inst_tran", "inst_tral", "inst_ipop", "inst_gena",
  "inst_gend", "inst_genr", "inst_geor", "inst_ris", "inst_jmp",
  "inst_bool", "inst_ift", "inst_iff", "inst_fork", "inst_jtry",
  "inst_ret", "inst_reta", "inst_retval", "inst_nop", "inst_ptry",
  "inst_end", "inst_swch", "inst_sele", "switch_list", "inst_once",
  "inst_band", "inst_bor", "inst_bxor", "inst_bnot", "inst_and", "inst_or",
  "inst_ands", "inst_ors", "inst_xors", "inst_mods", "inst_pows",
  "inst_nots", "inst_has", "inst_hasn", "inst_give", "inst_givn",
  "inst_in", "inst_noin", "inst_prov", "inst_psin", "inst_pass",
  "inst_shr", "inst_shl", "inst_shrs", "inst_shls", "inst_ldvr",
  "inst_ldpr", "inst_lsb", "inst_indi", "inst_stex", "inst_trac",
  "inst_wrt", "inst_sto", "inst_forb", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   161,   162,   162,   163,   163,   163,   163,   163,   163,
     164,   164,   165,   165,   166,   166,   167,   167,   167,   167,
     167,   167,   168,   168,   169,   169,   169,   169,   170,   170,
     170,   171,   171,   172,   172,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   174,   173,   173,   173,   173,   173,   173,
     173,   175,   175,   176,   176,   177,   178,   178,   178,   179,
     179,   180,   180,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   182,   182,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   188,   188,
     189,   189,   190,   190,   191,   191,   192,   192,   193,   193,
     194,   194,   195,   195,   196,   196,   197,   197,   198,   198,
     199,   199,   200,   200,   201,   201,   201,   202,   202,   203,
     203,   204,   204,   205,   205,   206,   206,   207,   207,   208,
     208,   208,   209,   209,   209,   210,   210,   211,   211,   212,
     212,   212,   213,   213,   214,   214,   215,   215,   216,   216,
     217,   217,   217,   218,   218,   218,   219,   219,   219,   220,
     220,   220,   221,   221,   221,   222,   222,   222,   223,   223,
     223,   224,   224,   224,   225,   225,   225,   226,   226,   226,
     227,   227,   227,   228,   228,   228,   229,   229,   229,   230,
     230,   231,   231,   232,   232,   233,   233,   234,   234,   235,
     235,   236,   236,   236,   237,   237,   238,   238,   238,   239,
     239,   239,   240,   240,   240,   241,   241,   241,   242,   242,
     243,   243,   244,   244,   245,   245,   246,   246,   247,   247,
     248,   248,   249,   249,   250,   250,   251,   251,   251,   252,
     252,   252,   252,   252,   253,   253,   253,   253,   253,   254,
     254,   254,   254,   254,   255,   255,   255,   256,   256,   257,
     257,   258,   258,   259,   259,   260,   260,   261,   261,   262,
     262,   263,   263,   264,   264,   264,   265,   265,   265,   266,
     266,   266,   267,   267,   267,   268,   268,   269,   269,   270,
     270,   271,   271,   272,   272,   273,   273,   274,   274,   275,
     275,   276,   276,   277,   277,   278,   278,   279,   279,   280,
     280,   280,   281,   281,   281,   282,   282,   283,   283,   284,
     284,   285,   285
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     4,     4,
       5,     3,     4,     3,     4,     3,     3,     2,     3,     3,
       4,     1,     2,     2,     3,     4,     4,     5,     5,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     6,     1,
       3,     3,     3,     2,     2,     4,     5,     3,     4,     2,
       3,     2,     1,     0,     4,     2,     3,     2,     2,     2,
       2,     1,     3,     1,     3,     2,     0,     1,     3,     1,
       1,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     4,     2,     4,     2,
       2,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     4,
       4,     2,     4,     4,     2,     4,     2,     4,     2,     2,
       1,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       4,     4,     2,     6,     6,     2,     6,     6,     2,     6,
       6,     2,     4,     4,     2,     4,     4,     2,     6,     6,
       2,     6,     6,     2,     6,     6,     2,     4,     4,     2,
       6,     6,     2,     6,     6,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     6,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     4,     4,     2,     4,
       4,     2,     4,     4,     2,     2,     2,     2,     1,     2,
       1,     2,     2,     2,     1,     2,     2,     2,     1,     2,
       6,     2,     6,     2,     3,     5,     4,     4,     2,     4,
       4,     4,     4,     2,     4,     4,     4,     4,     2,     4,
       4,     4,     4,     2,     2,     2,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     2,     2,     4,     4,     2,     4,     4,     2,     4,
       4,     2,     4,     4,     2,     4,     2,     4,     2,     4,
       2,     2,     2,     2,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     4,
       2,     4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    35,     0,     0,     0,     0,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
      82,     0,     0,     0,     0,     0,     0,     0,     0,    69,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   270,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     3,     0,     0,     0,   103,   168,   104,
     105,   111,   106,   112,   107,   113,   108,   114,   109,   110,
     119,   120,   121,   122,   123,   124,   162,   115,   116,   117,
     118,   125,   126,   156,   157,   154,   155,   127,   128,   129,
     190,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     140,   139,   141,   142,   143,   144,   145,   146,   147,   148,
     165,   149,   150,   151,   152,   153,   164,   158,   160,   159,
     161,   163,   169,   166,   167,   189,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   180,   181,   179,   182,   183,
     184,   185,   186,   187,   188,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   200,   201,   202,   203,   204,   205,
       9,   101,     0,     0,   101,   101,   101,    47,   101,     0,
       0,   101,    79,    81,    85,   101,    36,    52,    53,   101,
      14,    16,    17,    18,    19,    20,    21,     0,    15,     0,
       0,     0,     0,     0,     0,    87,    88,    89,    90,    91,
      73,    93,    74,    83,     0,    95,   207,     0,   211,   210,
     213,    30,    24,    29,    28,    25,    26,    10,     0,    11,
      12,    13,    27,   217,     0,   221,     0,   225,     0,   229,
       0,   231,     0,   215,     0,   219,     0,   223,     0,   227,
       0,   400,     0,   248,   247,   250,   249,   252,   251,   254,
     253,   256,   255,   258,   257,   349,   353,   352,   351,   344,
       0,   271,   269,   273,   272,   275,   274,   282,     0,   285,
       0,   288,     0,   291,     0,   294,     0,   297,     0,   300,
       0,   303,     0,   306,     0,   309,     0,   312,     0,     0,
     315,     0,     0,   318,   317,   316,   320,   319,   279,   278,
     322,   321,   324,   323,   326,     0,   328,   327,   333,   331,
     332,   338,     0,     0,   341,     0,     0,   335,   334,   233,
       0,   235,     0,   239,     0,   237,     0,   243,     0,   241,
       0,   266,     0,   268,     0,   261,     0,   264,     0,   361,
       0,   363,     0,   355,   246,   244,   245,   347,   345,   346,
     357,   356,   330,   329,   209,     0,   368,     0,     0,   373,
       0,     0,   378,     0,     0,   383,     0,     0,   386,   384,
     385,   398,     0,   388,     0,   390,     0,   392,     0,   394,
       0,   396,     0,   402,   401,   405,     0,   408,     0,   411,
       0,   414,     0,   416,     0,   418,     0,   420,     0,   359,
     277,   276,   422,   421,   424,   423,   426,     0,   428,     0,
     430,     0,   432,     0,   434,     0,   436,     0,   438,     0,
     441,    32,    31,   440,   439,   444,   443,   442,   446,   445,
     448,   447,   450,     0,   452,     0,     5,     7,     0,     8,
     102,    37,    22,   101,    23,    41,    43,    45,    46,    48,
      49,    71,    70,    72,    77,    80,    86,    54,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    96,   101,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,    38,    39,    42,    44,
      50,    78,    55,    56,    59,    60,    61,    62,    67,    64,
       0,    65,    66,    63,    92,    94,   100,    99,    84,    97,
      75,   206,   212,   216,   220,   224,   228,   230,   214,   218,
     222,   226,   399,   342,   343,   280,   281,     0,     0,     0,
       0,     0,     0,   292,   293,   295,   296,     0,     0,     0,
       0,     0,     0,   307,   308,     0,     0,     0,     0,     0,
     336,   337,   339,   340,   232,   234,   238,   236,   242,   240,
     265,   267,   260,   259,   263,   262,     0,     0,   208,   367,
     366,   369,   370,   371,   372,   374,   375,   376,   377,   379,
     380,   381,   382,   397,   387,   389,   391,   393,   395,   404,
     403,   407,   406,   410,   409,   413,   412,   415,   417,   419,
     425,   427,   429,   431,   433,   435,   437,   449,   451,    40,
      34,    33,    57,    58,     0,     0,    76,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    68,    98,   283,   284,
     286,   287,   289,   290,   298,   299,   301,   302,   304,   305,
     311,   310,   313,   314,   325,     0,   360,   362,     0,     0,
     364,     0,     0,   365
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   143,   308,   309,   310,   278,   657,   311,   312,
     514,   762,   144,   558,   290,   292,   145,   658,   659,   531,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168,   169,   170,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   200,   201,   202,   203,   204,   205,
     206,   207,   208,   209,   210,   211,   212,   213,   214,   806,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -245
static const yytype_int16 yypact[] =
{
    -245,   787,  -245,    35,  -245,  -245,   106,   167,   170,   179,
     185,   186,   215,  -245,   234,   236,   237,   266,   267,   196,
    -245,   284,   285,   288,     8,   289,  1104,  1104,   692,  -245,
     250,   243,   244,   246,   258,   259,   261,   264,   310,    97,
     479,    82,   159,   177,   198,   214,   232,   633,  1313,  1324,
    1339,  1354,  1365,  1375,  1389,  1404,   274,   290,     3,   464,
     119,     6,   556,   659,  -245,  1416,  1426,  1439,  1454,  1467,
    1477,  1489,  1504,  1518,  1528,  1539,   176,   253,   302,    17,
    1554,    19,    70,   537,   730,   303,   307,   308,   674,   715,
     947,   962,   977,  1003,  1018,  1569,   172,   273,   460,   463,
     486,   169,   309,   458,   487,  1033,  1580,   459,    72,   138,
     140,   147,  1590,  1059,  1074,  1604,  1619,  1631,  1641,  1654,
    1669,  1682,  1692,  1089,  1115,  1130,   233,  1704,  1719,  1733,
    1145,  1171,  1186,  1201,  1227,  1242,  1257,    29,   116,  1283,
    1298,  1743,     4,  -245,   330,   298,   346,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,   306,   906,   906,   306,   306,   306,   424,   306,   431,
     426,   306,   427,  -245,  -245,   306,  -245,  -245,  -245,   306,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   465,  -245,   466,
     468,   153,   469,   471,   472,  -245,  -245,  -245,  -245,  -245,
     473,  -245,   474,  -245,   477,  -245,  -245,   496,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   528,  -245,
    -245,  -245,  -245,  -245,   529,  -245,   535,  -245,   540,  -245,
     541,  -245,   553,  -245,   554,  -245,   557,  -245,   558,  -245,
     565,  -245,   566,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
     574,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   575,  -245,
     608,  -245,   628,  -245,   631,  -245,   632,  -245,   642,  -245,
     643,  -245,   644,  -245,   656,  -245,   657,  -245,   658,   660,
    -245,   664,   665,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   666,  -245,  -245,  -245,  -245,
    -245,  -245,   667,   668,  -245,   679,   680,  -245,  -245,  -245,
     681,  -245,   682,  -245,   693,  -245,   694,  -245,   696,  -245,
     697,  -245,   725,  -245,   729,  -245,   732,  -245,   733,  -245,
     734,  -245,   735,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   738,  -245,   741,   744,  -245,
     748,   749,  -245,   750,   752,  -245,   753,   767,  -245,  -245,
    -245,  -245,   781,  -245,   784,  -245,   786,  -245,   788,  -245,
     799,  -245,   823,  -245,  -245,  -245,   824,  -245,   830,  -245,
     831,  -245,   834,  -245,   835,  -245,   840,  -245,   944,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   946,  -245,   948,
    -245,   949,  -245,   950,  -245,   951,  -245,   982,  -245,   984,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,   985,  -245,  1038,  -245,  -245,   596,  -245,
    -245,   622,  -245,   306,  -245,   655,   910,  -245,  -245,  -245,
    1004,  -245,  -245,  -245,  1057,  -245,  -245,    12,    28,    55,
     634,   731,   517,  1041,  1097,  1098,   751,  1107,  1753,   306,
     921,   921,   921,   921,   921,   921,   921,   921,   921,   921,
     921,  1768,   105,  1783,  1783,  1783,  1783,  1783,  1783,  1783,
    1783,  1783,  1783,   921,   921,  1151,  1113,  1768,   921,   921,
     921,   921,   921,   921,   921,   921,   921,   921,  1104,  1104,
     519,   594,  1104,  1104,   921,   921,   921,    -8,    -7,    24,
      43,    44,    56,  1768,   921,   921,  1768,  1768,  1768,   571,
     936,   992,  1048,  1768,  1768,  1768,   921,   921,   921,   921,
     921,   921,   921,   921,   921,  -245,  -245,  1116,  -245,  -245,
    -245,  -245,    15,    15,  -245,  -245,  -245,  -245,  -245,  -245,
    1196,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  1199,  -245,
    1163,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  1202,  1203,  1204,
    1205,  1206,  1208,  -245,  -245,  -245,  -245,  1209,  1252,  1255,
    1256,  1258,  1259,  -245,  -245,  1260,  1261,  1262,  1264,  1265,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  1308,  1311,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1314,  1753,  -245,  1104,  1104,   921,
     921,   921,   921,  1104,  1104,   921,   921,  1104,  1104,   921,
     921,  1162,  1218,   906,  1273,  1273,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  1315,  1316,  1316,  1318,  1275,
    -245,  1321,  1319,  -245
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -245,  -245,  -245,    62,   -81,   -26,  -245,  -242,  -244,   -82,
    -107,  -175,  -245,  -245,  -245,  -245,  -245,  -245,   592,  -203,
    1219,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   582,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -359
static const yytype_int16 yytable[] =
{
     277,   279,   395,   397,   345,   524,  -348,   349,   534,   534,
     533,   535,   267,   297,   299,   534,   642,   542,   386,   760,
     390,   324,   326,   328,   330,   332,   334,   336,   338,   340,
     510,   517,   644,   721,   723,   525,   722,   724,   250,   356,
     358,   360,   362,   364,   366,   368,   370,   372,   374,   376,
     350,   536,   537,   538,   389,   540,   511,   512,   544,   646,
     268,   387,   546,   391,   643,   725,   547,   761,   726,   422,
     270,   392,   645,   449,   271,   272,   273,   274,   275,   276,
     445,   511,   512,   300,   727,   729,   462,   728,   730,   468,
     470,   472,   474,   476,   478,   480,   482,   731,   296,   647,
     732,   491,   493,   495,   314,   316,   318,   320,   322,   673,
     251,   513,   516,   450,   393,   523,   451,   515,   342,   344,
     348,   347,  -350,   270,   352,   354,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   452,
     307,   455,   271,   272,   273,   274,   275,   276,   458,   674,
     408,   410,   412,   414,   416,   418,   420,   270,   551,   552,
     313,   271,   272,   273,   274,   275,   276,   443,   511,   512,
     433,   252,  -354,   423,   253,   464,   466,   377,   315,   453,
     378,   456,   454,   254,   457,   484,   486,   488,   459,   255,
     256,   460,   497,   499,   501,   503,   505,   507,   509,   317,
     270,   519,   521,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   319,   424,   307,   270,   257,
     379,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   321,   489,   307,  -358,   263,   258,   270,
     259,   260,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   380,   270,   307,   381,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     261,   262,   307,   270,   425,   341,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   264,   265,
     307,   343,   266,   269,   285,   286,   287,   382,   288,   289,
     291,   527,   293,   383,   398,   294,   384,   399,   401,   404,
     434,   402,   405,   435,   534,   270,   295,   426,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     637,   270,   307,   526,   301,   271,   272,   273,   274,   275,
     276,   302,   303,   304,   305,   306,   385,   400,   307,   529,
     530,   403,   406,   436,    39,    40,   660,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   437,
     446,   427,   438,   447,   429,   346,   539,   543,   763,   545,
     548,   549,   541,   550,   553,   301,   554,   555,   556,   557,
     298,   559,   302,   303,   304,   305,   306,   431,   440,   532,
     672,   676,   678,   680,   682,   684,   686,   688,   690,   692,
     694,   560,   439,   448,   428,   270,   699,   430,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     270,   534,   307,   712,   271,   272,   273,   274,   275,   276,
     432,   441,   733,   561,   562,   736,   737,   738,   394,   534,
     563,   804,   747,   748,   749,   564,   565,   675,   677,   679,
     681,   683,   685,   687,   689,   691,   693,   351,   566,   567,
     270,   650,   568,   569,   271,   272,   273,   274,   275,   276,
     570,   571,   710,   711,   713,   715,   716,   717,   270,   572,
     573,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   740,   742,   744,   746,   270,   714,   635,
     301,   271,   272,   273,   274,   275,   276,   302,   303,   304,
     305,   306,   270,   574,   307,   739,   271,   272,   273,   274,
     275,   276,   661,   662,   663,   664,   665,   666,   667,   668,
     669,   670,   671,   575,   323,   270,   576,   577,   648,   271,
     272,   273,   274,   275,   276,   695,   696,   578,   579,   580,
     700,   701,   702,   703,   704,   705,   706,   707,   708,   709,
     353,   581,   582,   583,   636,   584,   718,   719,   720,   585,
     586,   587,   588,   589,   270,   407,   734,   735,   271,   272,
     273,   274,   275,   276,   590,   591,   592,   593,   750,   751,
     752,   753,   754,   755,   756,   757,   758,   638,   594,   595,
     270,   596,   597,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   270,   409,   307,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     598,   396,   307,   280,   599,   649,   281,   600,   601,   602,
     603,   788,   789,   604,   282,   283,   605,   794,   795,   606,
     284,   798,   799,   607,   608,   609,   270,   610,   611,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   270,   612,   307,   301,   271,   272,   273,   274,   275,
     276,   302,   303,   304,   305,   306,   613,     2,     3,   614,
       4,   615,   654,   616,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,   617,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,   618,   619,
      38,   790,   791,   792,   793,   620,   621,   796,   797,   622,
     623,   800,   801,    39,    40,   624,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   411,   625,
     301,   626,   639,   627,   628,   629,   630,   302,   303,   304,
     305,   306,   270,   413,   532,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   270,   415,   307,
     741,   271,   272,   273,   274,   275,   276,   631,   270,   632,
     633,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   270,   417,   307,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   419,
     307,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   270,   442,   307,   743,   271,   272,   273,
     274,   275,   276,   634,   270,   651,   640,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   270,
     463,   307,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   270,   465,   307,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   270,
     483,   307,   745,   271,   272,   273,   274,   275,   276,   641,
     270,   652,   653,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   270,   485,   307,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     270,   487,   307,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   270,   496,   307,   655,   271,
     272,   273,   274,   275,   276,   697,   270,   698,   759,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   270,   498,   307,   301,   271,   272,   273,   274,   275,
     276,   302,   303,   304,   305,   306,   270,   500,   307,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   764,   502,   307,   765,   766,   802,   767,   768,   769,
     770,   771,   270,   772,   773,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   270,   504,   307,
     301,   271,   272,   273,   274,   275,   276,   302,   303,   304,
     305,   306,   270,   506,   307,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   774,   508,   307,
     775,   776,   803,   777,   778,   779,   780,   781,   270,   782,
     783,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   270,   518,   307,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   520,
     307,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   784,   325,   307,   785,   805,   786,   811,
     808,   809,   810,   813,   270,   327,   812,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   270,
     329,   307,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   270,   331,   307,   787,   271,   272,
     273,   274,   275,   276,   528,   270,   333,   807,     0,   271,
     272,   273,   274,   275,   276,     0,   335,     0,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     337,     0,     0,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   339,   270,     0,     0,     0,
     271,   272,   273,   274,   275,   276,   270,   355,     0,     0,
     271,   272,   273,   274,   275,   276,     0,   357,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     359,     0,     0,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   361,     0,   270,     0,     0,
       0,   271,   272,   273,   274,   275,   276,   270,   363,     0,
       0,   271,   272,   273,   274,   275,   276,     0,   365,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     367,     0,     0,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   369,     0,     0,   270,     0,
       0,     0,   271,   272,   273,   274,   275,   276,   270,   371,
       0,     0,   271,   272,   273,   274,   275,   276,     0,   373,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     375,     0,     0,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   388,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   270,
     421,     0,     0,   271,   272,   273,   274,   275,   276,     0,
     270,   444,     0,     0,   271,   272,   273,   274,   275,   276,
       0,   461,     0,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   467,     0,     0,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     469,   270,     0,     0,     0,   271,   272,   273,   274,   275,
     276,   270,   471,     0,     0,   271,   272,   273,   274,   275,
     276,     0,   473,     0,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   475,     0,     0,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     477,     0,   270,     0,     0,     0,   271,   272,   273,   274,
     275,   276,   270,   479,     0,     0,   271,   272,   273,   274,
     275,   276,     0,   481,     0,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   490,     0,     0,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     492,     0,     0,   270,     0,     0,     0,   271,   272,   273,
     274,   275,   276,   270,   494,     0,     0,   271,   272,   273,
     274,   275,   276,     0,   522,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,     0,     0,     0,     0,     0,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
       0,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   656,     0,     0,   301,     0,     0,
       0,     0,     0,     0,   302,   303,   304,   305,   306,   270,
       0,   532,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   270,     0,     0,   301,   271,   272,
     273,   274,   275,   276,     0,   303,   304
};

static const yytype_int16 yycheck[] =
{
      26,    27,    83,    84,     1,     1,     3,     1,   252,   253,
     252,   253,     4,    39,    40,   259,     4,   259,     1,     4,
       1,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       1,   138,     4,    41,    41,   142,    44,    44,     3,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      44,   254,   255,   256,    80,   258,    52,    53,   261,     4,
      52,    44,   265,    44,    52,    41,   269,    52,    44,    95,
      41,     1,    44,     1,    45,    46,    47,    48,    49,    50,
     106,    52,    53,     1,    41,    41,   112,    44,    44,   115,
     116,   117,   118,   119,   120,   121,   122,    41,     1,    44,
      44,   127,   128,   129,    42,    43,    44,    45,    46,     4,
       4,   137,   138,    41,    44,   141,    44,     1,    56,    57,
       1,    59,     3,    41,    62,    63,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,     1,    45,    46,    47,    48,    49,    50,     1,    44,
      88,    89,    90,    91,    92,    93,    94,    41,     5,     6,
       1,    45,    46,    47,    48,    49,    50,   105,    52,    53,
       1,     4,     3,     1,     4,   113,   114,     1,     1,    41,
       4,    41,    44,     4,    44,   123,   124,   125,    41,     4,
       4,    44,   130,   131,   132,   133,   134,   135,   136,     1,
      41,   139,   140,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     1,    44,    58,    41,     4,
      44,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     1,     1,    58,     3,    41,     4,    41,
       4,     4,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,     1,    41,    58,     4,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       4,     4,    58,    41,     1,     1,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,     4,     4,
      58,     1,     4,     4,    44,    52,    52,    44,    52,    41,
      41,     3,    41,     1,     1,    41,     4,     4,     1,     1,
       1,     4,     4,     4,   558,    41,     6,    44,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
     533,    41,    58,     3,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    44,    44,    58,     3,
      44,    44,    44,    44,    56,    57,   559,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,     1,
       1,     1,     4,     4,     1,     1,    42,    41,   643,    42,
       5,     5,    41,     5,     5,    44,     5,     5,     5,     5,
       1,     4,    51,    52,    53,    54,    55,     1,     1,    58,
     571,   573,   574,   575,   576,   577,   578,   579,   580,   581,
     582,     5,    44,    44,    44,    41,   587,    44,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,   765,    58,     4,    45,    46,    47,    48,    49,    50,
      44,    44,   613,     5,     5,   616,   617,   618,     1,   783,
       5,   783,   623,   624,   625,     5,     5,   573,   574,   575,
     576,   577,   578,   579,   580,   581,   582,     1,     5,     5,
      41,    44,     5,     5,    45,    46,    47,    48,    49,    50,
       5,     5,   598,   599,   600,   601,   602,   603,    41,     5,
       5,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,   619,   620,   621,   622,    41,     4,     3,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    41,     5,    58,    44,    45,    46,    47,    48,
      49,    50,   560,   561,   562,   563,   564,   565,   566,   567,
     568,   569,   570,     5,     1,    41,     5,     5,     4,    45,
      46,    47,    48,    49,    50,   583,   584,     5,     5,     5,
     588,   589,   590,   591,   592,   593,   594,   595,   596,   597,
       1,     5,     5,     5,    42,     5,   604,   605,   606,     5,
       5,     5,     5,     5,    41,     1,   614,   615,    45,    46,
      47,    48,    49,    50,     5,     5,     5,     5,   626,   627,
     628,   629,   630,   631,   632,   633,   634,    42,     5,     5,
      41,     5,     5,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    58,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       5,     1,    58,    41,     5,     4,    44,     5,     5,     5,
       5,   767,   768,     5,    52,    53,     5,   773,   774,     5,
      58,   777,   778,     5,     5,     5,    41,     5,     5,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    41,     5,    58,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,     5,     0,     1,     5,
       3,     5,    41,     5,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     5,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,     5,     5,
      43,   769,   770,   771,   772,     5,     5,   775,   776,     5,
       5,   779,   780,    56,    57,     5,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,     1,     5,
      44,     5,    42,     5,     5,     5,     5,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,     5,    41,     5,
       5,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,     5,    41,     4,    42,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    42,
      41,     4,     4,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    58,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,     1,    58,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    58,    41,    45,
      46,    47,    48,    49,    50,     4,    41,    44,    42,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    41,     1,    58,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    41,     1,    58,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,     5,     1,    58,     5,    42,    44,     5,     5,     5,
       5,     5,    41,     5,     5,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,     5,     1,    58,
       5,     5,    44,     5,     5,     5,     5,     5,    41,     5,
       5,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     5,     1,    58,     5,    44,     4,    44,
       5,     5,     4,     4,    41,     1,     5,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,   765,    45,    46,
      47,    48,    49,    50,   145,    41,     1,   785,    -1,    45,
      46,    47,    48,    49,    50,    -1,     1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    41,    -1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    41,     1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    -1,     1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    41,    -1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    41,     1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    -1,     1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    -1,    41,    -1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    41,     1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    -1,     1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,    41,
       1,    -1,    -1,    45,    46,    47,    48,    49,    50,    -1,
      41,     1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      -1,     1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    41,    -1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    41,     1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    -1,     1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    41,     1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    -1,     1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
       1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    41,     1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    -1,     1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    -1,    -1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    41,    -1,    -1,    44,    -1,    -1,
      -1,    -1,    -1,    -1,    51,    52,    53,    54,    55,    41,
      -1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,    -1,    -1,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    53
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   162,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    43,    56,
      57,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   163,   173,   177,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       3,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,    41,     4,     4,     4,     4,    52,     4,
      41,    45,    46,    47,    48,    49,    50,   166,   167,   166,
      41,    44,    52,    53,    58,    44,    52,    52,    52,    41,
     175,    41,   176,    41,    41,     6,     1,   166,     1,   166,
       1,    44,    51,    52,    53,    54,    55,    58,   164,   165,
     166,   169,   170,     1,   164,     1,   164,     1,   164,     1,
     164,     1,   164,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   164,     1,   164,     1,     1,   164,     1,     1,
      44,     1,   164,     1,   164,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,     4,    44,
       1,     4,    44,     1,     4,    44,     1,    44,     1,   166,
       1,    44,     1,    44,     1,   165,     1,   165,     1,     4,
      44,     1,     4,    44,     1,     4,    44,     1,   164,     1,
     164,     1,   164,     1,   164,     1,   164,     1,   164,     1,
     164,     1,   166,     1,    44,     1,    44,     1,    44,     1,
      44,     1,    44,     1,     1,     4,    44,     1,     4,    44,
       1,    44,     1,   164,     1,   166,     1,     4,    44,     1,
      41,    44,     1,    41,    44,     1,    41,    44,     1,    41,
      44,     1,   166,     1,   164,     1,   164,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   164,     1,   164,     1,   164,     1,
       1,   166,     1,   166,     1,   166,     1,   164,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   164,
       1,    52,    53,   166,   171,     1,   166,   171,     1,   164,
       1,   164,     1,   166,     1,   171,     3,     3,   181,     3,
      44,   180,    58,   168,   169,   168,   180,   180,   180,    42,
     180,    41,   168,    41,   180,    42,   180,   180,     5,     5,
       5,     5,     6,     5,     5,     5,     5,     5,   174,     4,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     3,    42,   180,    42,    42,
      42,    42,     4,    52,     4,    44,     4,    44,     4,     4,
      44,     4,     4,     4,    41,    41,    41,   168,   178,   179,
     180,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   165,     4,    44,   166,   170,   166,   170,   166,
     170,   166,   170,   166,   170,   166,   170,   166,   170,   166,
     170,   166,   170,   166,   170,   164,   164,     4,    44,   165,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     166,   166,     4,   166,     4,   166,   166,   166,   164,   164,
     164,    41,    44,    41,    44,    41,    44,    41,    44,    41,
      44,    41,    44,   165,   164,   164,   165,   165,   165,    44,
     166,    44,   166,    44,   166,    44,   166,   165,   165,   165,
     164,   164,   164,   164,   164,   164,   164,   164,   164,    42,
       4,    52,   172,   172,     5,     5,    42,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     4,   179,   166,   166,
     164,   164,   164,   164,   166,   166,   164,   164,   166,   166,
     164,   164,    44,    44,   168,    44,   250,   250,     5,     5,
       4,    44,     5,     4
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 9:
#line 242 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 35:
#line 258 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 36:
#line 259 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 37:
#line 260 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 38:
#line 261 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 39:
#line 262 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 40:
#line 263 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 41:
#line 264 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 265 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 266 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 267 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 45:
#line 268 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 46:
#line 269 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 47:
#line 270 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 48:
#line 271 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 49:
#line 272 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 50:
#line 273 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 51:
#line 274 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 52:
#line 275 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), false ); }
    break;

  case 53:
#line 276 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), true ); }
    break;

  case 54:
#line 277 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 55:
#line 278 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, false ); }
    break;

  case 56:
#line 279 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, true ); }
    break;

  case 57:
#line 280 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 58:
#line 283 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 59:
#line 286 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 60:
#line 287 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 61:
#line 288 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 62:
#line 289 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 63:
#line 290 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 64:
#line 291 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 65:
#line 292 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 66:
#line 293 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 67:
#line 294 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 68:
#line 295 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 69:
#line 296 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 70:
#line 297 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 71:
#line 298 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 72:
#line 299 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 75:
#line 302 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 76:
#line 303 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 77:
#line 304 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 78:
#line 305 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 79:
#line 306 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 80:
#line 307 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 81:
#line 308 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 82:
#line 309 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 83:
#line 310 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 85:
#line 311 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 86:
#line 312 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 87:
#line 313 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 88:
#line 315 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 89:
#line 320 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 90:
#line 326 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 91:
#line 334 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 92:
#line 335 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 93:
#line 339 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 94:
#line 340 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 95:
#line 343 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 97:
#line 347 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); }
    break;

  case 98:
#line 348 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); }
    break;

  case 101:
#line 357 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 206:
#line 468 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 207:
#line 469 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 208:
#line 473 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 209:
#line 474 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 210:
#line 478 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 211:
#line 479 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 212:
#line 483 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 213:
#line 484 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 214:
#line 488 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 215:
#line 489 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 216:
#line 494 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 217:
#line 495 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 218:
#line 499 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 219:
#line 500 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 220:
#line 504 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 221:
#line 505 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 222:
#line 509 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 223:
#line 510 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 224:
#line 515 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 225:
#line 516 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 226:
#line 520 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 227:
#line 521 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 228:
#line 525 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 229:
#line 526 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 230:
#line 530 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 231:
#line 531 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 232:
#line 536 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 233:
#line 537 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 234:
#line 541 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 235:
#line 542 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 236:
#line 546 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 237:
#line 547 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 238:
#line 551 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 239:
#line 552 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 240:
#line 556 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 241:
#line 557 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 242:
#line 561 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 243:
#line 562 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 244:
#line 566 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 245:
#line 567 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 246:
#line 568 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 247:
#line 572 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 248:
#line 573 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 249:
#line 577 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 250:
#line 578 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 251:
#line 583 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 252:
#line 584 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 253:
#line 588 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 254:
#line 589 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 255:
#line 594 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 256:
#line 595 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 257:
#line 599 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 258:
#line 600 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 259:
#line 604 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 260:
#line 605 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 261:
#line 606 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 262:
#line 610 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 263:
#line 611 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 264:
#line 612 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 265:
#line 616 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 266:
#line 617 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 267:
#line 621 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 268:
#line 622 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 269:
#line 627 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 270:
#line 628 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 271:
#line 629 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 272:
#line 633 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 273:
#line 634 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 274:
#line 639 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 275:
#line 640 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 276:
#line 644 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 277:
#line 645 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 278:
#line 649 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 279:
#line 650 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 280:
#line 655 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 281:
#line 656 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 282:
#line 657 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 283:
#line 661 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 284:
#line 662 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 285:
#line 663 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 286:
#line 667 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 287:
#line 668 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 288:
#line 669 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 289:
#line 673 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 290:
#line 674 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 291:
#line 675 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 292:
#line 679 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 293:
#line 680 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 294:
#line 681 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 295:
#line 685 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 296:
#line 686 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 297:
#line 687 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 298:
#line 691 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 299:
#line 692 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 300:
#line 693 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 301:
#line 697 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 302:
#line 698 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 303:
#line 699 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 304:
#line 703 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 305:
#line 704 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 306:
#line 705 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 307:
#line 709 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 308:
#line 710 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 309:
#line 711 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 310:
#line 715 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 311:
#line 716 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 312:
#line 717 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 313:
#line 721 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 314:
#line 722 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 315:
#line 723 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 316:
#line 727 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 317:
#line 728 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 318:
#line 729 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 319:
#line 733 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 320:
#line 734 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 321:
#line 738 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 322:
#line 739 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 323:
#line 743 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 324:
#line 744 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 325:
#line 748 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 326:
#line 749 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 327:
#line 753 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 328:
#line 754 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 329:
#line 758 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 330:
#line 759 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 331:
#line 763 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 332:
#line 764 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 333:
#line 765 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 334:
#line 769 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 335:
#line 770 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 336:
#line 774 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 337:
#line 775 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 338:
#line 776 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 339:
#line 780 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 340:
#line 781 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 341:
#line 782 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 342:
#line 787 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 343:
#line 788 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 344:
#line 789 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 345:
#line 793 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 346:
#line 794 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 347:
#line 795 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 348:
#line 799 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 349:
#line 800 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 350:
#line 804 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 351:
#line 805 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 352:
#line 809 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 353:
#line 810 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 354:
#line 814 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 355:
#line 815 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 356:
#line 819 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 357:
#line 820 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 358:
#line 824 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 359:
#line 825 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 360:
#line 829 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 361:
#line 830 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 362:
#line 834 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 363:
#line 835 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 364:
#line 840 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 365:
#line 849 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 366:
#line 857 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 367:
#line 858 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 368:
#line 859 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 369:
#line 863 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 370:
#line 864 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 371:
#line 865 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 372:
#line 866 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 373:
#line 867 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 374:
#line 871 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 375:
#line 872 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 376:
#line 873 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 377:
#line 874 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 378:
#line 875 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 379:
#line 879 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 380:
#line 880 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 381:
#line 881 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 382:
#line 882 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 383:
#line 883 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 384:
#line 887 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 385:
#line 888 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 386:
#line 889 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 387:
#line 893 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 388:
#line 894 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 389:
#line 898 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 390:
#line 899 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 391:
#line 903 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 392:
#line 904 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 393:
#line 908 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 394:
#line 909 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 395:
#line 913 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 396:
#line 914 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 397:
#line 918 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 398:
#line 919 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 399:
#line 923 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 400:
#line 924 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 401:
#line 928 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 402:
#line 929 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 403:
#line 933 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 404:
#line 934 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 405:
#line 935 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 406:
#line 939 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 407:
#line 940 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 408:
#line 941 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 409:
#line 945 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 410:
#line 946 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 411:
#line 947 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 412:
#line 951 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 413:
#line 952 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 414:
#line 953 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 415:
#line 958 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 416:
#line 959 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 417:
#line 963 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 418:
#line 964 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 419:
#line 968 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 420:
#line 969 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 421:
#line 973 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 422:
#line 974 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 423:
#line 978 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 424:
#line 979 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 425:
#line 983 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 426:
#line 984 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 427:
#line 988 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 428:
#line 989 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 429:
#line 993 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 430:
#line 994 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 431:
#line 998 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 432:
#line 999 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 433:
#line 1003 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 434:
#line 1004 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 435:
#line 1008 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 436:
#line 1009 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 437:
#line 1013 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 438:
#line 1014 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 439:
#line 1018 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 440:
#line 1019 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 441:
#line 1020 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 442:
#line 1024 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 443:
#line 1025 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 444:
#line 1026 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 445:
#line 1030 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 446:
#line 1031 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 447:
#line 1035 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 448:
#line 1036 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 449:
#line 1041 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 450:
#line 1042 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;

  case 451:
#line 1046 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_FORB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 452:
#line 1047 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORB" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4262 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1050 "/Users/gniccolai/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


