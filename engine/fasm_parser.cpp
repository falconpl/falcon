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
     I_STO = 414
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




/* Copy the first part of user declarations.  */
#line 17 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"

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
#line 470 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"

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
# if YYENABLE_NLS
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
#define YYLAST   1843

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  160
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  124
/* YYNRULES -- Number of rules.  */
#define YYNRULES  446
/* YYNRULES -- Number of states.  */
#define YYNSTATES  805

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   414

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
     155,   156,   157,   158,   159
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
     137,   142,   144,   147,   151,   156,   162,   167,   172,   177,
     182,   187,   192,   197,   202,   207,   214,   216,   220,   224,
     228,   231,   234,   239,   245,   249,   254,   257,   261,   264,
     266,   267,   272,   275,   279,   282,   285,   288,   291,   293,
     297,   299,   303,   306,   307,   309,   313,   315,   317,   318,
     320,   322,   324,   326,   328,   330,   332,   334,   336,   338,
     340,   342,   344,   346,   348,   350,   352,   354,   356,   358,
     360,   362,   364,   366,   368,   370,   372,   374,   376,   378,
     380,   382,   384,   386,   388,   390,   392,   394,   396,   398,
     400,   402,   404,   406,   408,   410,   412,   414,   416,   418,
     420,   422,   424,   426,   428,   430,   432,   434,   436,   438,
     440,   442,   444,   446,   448,   450,   452,   454,   456,   458,
     460,   462,   464,   466,   468,   470,   472,   474,   476,   478,
     480,   482,   484,   486,   488,   490,   492,   494,   496,   498,
     500,   502,   504,   506,   508,   510,   512,   514,   516,   518,
     520,   522,   524,   529,   532,   537,   540,   543,   546,   551,
     554,   559,   562,   567,   570,   575,   578,   583,   586,   591,
     594,   599,   602,   607,   610,   615,   618,   623,   626,   631,
     634,   639,   642,   647,   650,   655,   658,   663,   666,   671,
     674,   677,   680,   683,   686,   689,   692,   695,   698,   701,
     704,   707,   710,   713,   716,   719,   724,   729,   732,   737,
     742,   745,   750,   753,   758,   761,   764,   766,   769,   772,
     775,   778,   781,   784,   787,   790,   793,   798,   803,   806,
     813,   820,   823,   830,   837,   840,   847,   854,   857,   862,
     867,   870,   875,   880,   883,   890,   897,   900,   907,   914,
     917,   924,   931,   934,   939,   944,   947,   954,   961,   964,
     971,   978,   981,   984,   987,   990,   993,   996,   999,  1002,
    1005,  1008,  1015,  1018,  1021,  1024,  1027,  1030,  1033,  1036,
    1039,  1042,  1045,  1050,  1055,  1058,  1063,  1068,  1071,  1076,
    1081,  1084,  1087,  1090,  1093,  1095,  1098,  1100,  1103,  1106,
    1109,  1111,  1114,  1117,  1120,  1122,  1125,  1132,  1135,  1142,
    1145,  1149,  1155,  1160,  1165,  1168,  1173,  1178,  1183,  1188,
    1191,  1196,  1201,  1206,  1211,  1214,  1219,  1224,  1229,  1234,
    1237,  1240,  1243,  1246,  1251,  1254,  1259,  1262,  1267,  1270,
    1275,  1278,  1283,  1286,  1291,  1294,  1299,  1302,  1305,  1308,
    1313,  1318,  1321,  1326,  1331,  1334,  1339,  1344,  1347,  1352,
    1357,  1360,  1365,  1368,  1373,  1376,  1381,  1384,  1387,  1390,
    1393,  1396,  1401,  1404,  1409,  1412,  1417,  1420,  1425,  1428,
    1433,  1436,  1441,  1444,  1449,  1452,  1455,  1458,  1461,  1464,
    1467,  1470,  1473,  1476,  1479,  1482,  1487
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     161,     0,    -1,    -1,   161,   162,    -1,     3,    -1,   172,
       3,    -1,   176,   180,     3,    -1,   176,     3,    -1,   180,
       3,    -1,     1,     3,    -1,    58,    -1,   164,    -1,   165,
      -1,   168,    -1,    41,    -1,   166,    -1,    45,    -1,    46,
      -1,    47,    -1,    48,    -1,    49,    -1,    50,    -1,    58,
      -1,   168,    -1,    51,    -1,    54,    -1,    55,    -1,   169,
      -1,    53,    -1,    52,    -1,    44,    -1,    53,    -1,    52,
      -1,    52,    -1,     4,    -1,     7,    -1,    26,     4,    -1,
       8,     4,   179,    -1,     8,     4,   179,    42,    -1,     9,
       4,   167,   179,    -1,     9,     4,   167,   179,    42,    -1,
      10,     4,   167,    -1,    10,     4,   167,    42,    -1,    11,
       4,   179,    -1,    11,     4,   179,    42,    -1,    12,     4,
     179,    -1,    13,     4,   179,    -1,    14,     4,    -1,    14,
       4,    42,    -1,    16,     4,   179,    -1,    16,     4,   179,
      42,    -1,    15,    -1,    27,     4,    -1,    28,     4,   179,
      -1,    28,     4,   179,   171,    -1,    28,     4,   179,   171,
     171,    -1,    29,   165,     5,     4,    -1,    29,   165,     5,
      44,    -1,    30,   165,     5,     4,    -1,    30,   165,     5,
      44,    -1,    31,    58,     5,     4,    -1,    31,    44,     5,
       4,    -1,    31,    52,     5,     4,    -1,    31,    53,     5,
       4,    -1,    31,    41,     5,     4,    -1,    31,    44,     6,
      44,     5,     4,    -1,    32,    -1,    18,     4,   167,    -1,
      18,     4,    41,    -1,    19,     4,    41,    -1,    37,   174,
      -1,    38,   175,    -1,    40,    41,     4,   179,    -1,    40,
      41,     4,   179,    42,    -1,    20,     4,   179,    -1,    20,
       4,   179,    42,    -1,    21,     4,    -1,    21,     4,    42,
      -1,    22,    41,    -1,    23,    -1,    -1,    39,    41,   173,
     177,    -1,    24,     4,    -1,    25,     4,   179,    -1,    33,
      44,    -1,    34,    52,    -1,    35,    52,    -1,    36,    52,
      -1,    41,    -1,   174,     5,    41,    -1,    41,    -1,   175,
       5,    41,    -1,    43,     6,    -1,    -1,   178,    -1,   177,
       5,   178,    -1,   167,    -1,    41,    -1,    -1,    44,    -1,
     181,    -1,   183,    -1,   184,    -1,   186,    -1,   188,    -1,
     190,    -1,   192,    -1,   193,    -1,   185,    -1,   187,    -1,
     189,    -1,   191,    -1,   201,    -1,   202,    -1,   203,    -1,
     204,    -1,   194,    -1,   195,    -1,   196,    -1,   197,    -1,
     198,    -1,   199,    -1,   205,    -1,   206,    -1,   211,    -1,
     212,    -1,   213,    -1,   215,    -1,   216,    -1,   217,    -1,
     218,    -1,   219,    -1,   220,    -1,   221,    -1,   222,    -1,
     223,    -1,   225,    -1,   224,    -1,   226,    -1,   227,    -1,
     228,    -1,   229,    -1,   230,    -1,   231,    -1,   232,    -1,
     233,    -1,   235,    -1,   236,    -1,   237,    -1,   238,    -1,
     239,    -1,   209,    -1,   210,    -1,   207,    -1,   208,    -1,
     241,    -1,   243,    -1,   242,    -1,   244,    -1,   200,    -1,
     245,    -1,   240,    -1,   234,    -1,   247,    -1,   248,    -1,
     182,    -1,   246,    -1,   251,    -1,   252,    -1,   253,    -1,
     254,    -1,   255,    -1,   256,    -1,   257,    -1,   258,    -1,
     259,    -1,   262,    -1,   260,    -1,   261,    -1,   263,    -1,
     264,    -1,   265,    -1,   266,    -1,   267,    -1,   268,    -1,
     269,    -1,   250,    -1,   214,    -1,   270,    -1,   271,    -1,
     272,    -1,   273,    -1,   274,    -1,   275,    -1,   276,    -1,
     277,    -1,   278,    -1,   279,    -1,   280,    -1,   281,    -1,
     282,    -1,   283,    -1,    56,   165,     5,   163,    -1,    56,
       1,    -1,   124,   165,     5,   163,    -1,   124,     1,    -1,
      57,   165,    -1,    57,     1,    -1,    59,   163,     5,   163,
      -1,    59,     1,    -1,    65,   165,     5,   163,    -1,    65,
       1,    -1,    60,   163,     5,   163,    -1,    60,     1,    -1,
      66,   165,     5,   163,    -1,    66,     1,    -1,    61,   163,
       5,   163,    -1,    61,     1,    -1,    67,   165,     5,   163,
      -1,    67,     1,    -1,    62,   163,     5,   163,    -1,    62,
       1,    -1,    68,   165,     5,   163,    -1,    68,     1,    -1,
      63,   163,     5,   163,    -1,    63,     1,    -1,    64,   163,
       5,   163,    -1,    64,     1,    -1,   107,   163,     5,   163,
      -1,   107,     1,    -1,   108,   163,     5,   163,    -1,   108,
       1,    -1,   110,   163,     5,   163,    -1,   110,     1,    -1,
     109,   163,     5,   163,    -1,   109,     1,    -1,   112,   163,
       5,   163,    -1,   112,     1,    -1,   111,   163,     5,   163,
      -1,   111,     1,    -1,   120,     4,    -1,   120,    44,    -1,
     120,     1,    -1,    70,   165,    -1,    70,     1,    -1,    71,
     165,    -1,    71,     1,    -1,    72,   165,    -1,    72,     1,
      -1,    73,   165,    -1,    73,     1,    -1,    74,   163,    -1,
      74,     1,    -1,    75,   163,    -1,    75,     1,    -1,   115,
      44,     5,   165,    -1,   115,    44,     5,     4,    -1,   115,
       1,    -1,   116,    44,     5,   165,    -1,   116,    44,     5,
       4,    -1,   116,     1,    -1,   113,   165,     5,   165,    -1,
     113,     1,    -1,   114,    44,     5,   165,    -1,   114,     1,
      -1,    80,   163,    -1,    82,    -1,    80,     1,    -1,    81,
     163,    -1,    81,     1,    -1,    83,   165,    -1,    83,     1,
      -1,   145,   165,    -1,   145,     1,    -1,    98,   165,    -1,
      98,     1,    -1,    84,   165,     5,   165,    -1,    84,   165,
       5,   169,    -1,    84,     1,    -1,    85,   165,     5,   165,
       5,   165,    -1,    85,   165,     5,   169,     5,   165,    -1,
      85,     1,    -1,    86,   165,     5,   165,     5,   163,    -1,
      86,   165,     5,   169,     5,   163,    -1,    86,     1,    -1,
      87,   165,     5,   165,     5,   163,    -1,    87,   165,     5,
     169,     5,   163,    -1,    87,     1,    -1,    88,   165,     5,
     165,    -1,    88,   165,     5,   169,    -1,    88,     1,    -1,
      89,   165,     5,   165,    -1,    89,   165,     5,   169,    -1,
      89,     1,    -1,    90,   165,     5,   165,     5,   165,    -1,
      90,   165,     5,   169,     5,   165,    -1,    90,     1,    -1,
      91,   165,     5,   165,     5,   163,    -1,    91,   165,     5,
     169,     5,   163,    -1,    91,     1,    -1,    92,   165,     5,
     165,     5,   165,    -1,    92,   165,     5,   169,     5,   165,
      -1,    92,     1,    -1,    93,   165,     5,   165,    -1,    93,
     165,     5,   169,    -1,    93,     1,    -1,    94,    44,     5,
     163,     5,   163,    -1,    94,     4,     5,   163,     5,   163,
      -1,    94,     1,    -1,    95,     4,     5,     4,     5,    44,
      -1,    95,    44,     5,    44,     5,    44,    -1,    95,     1,
      -1,    96,    44,    -1,    96,     4,    -1,    96,     1,    -1,
      97,    44,    -1,    97,     1,    -1,    99,    44,    -1,    99,
       1,    -1,   100,    44,    -1,   100,     1,    -1,   101,   164,
       5,   164,     5,   167,    -1,   101,     1,    -1,   102,   164,
      -1,   102,     1,    -1,   123,   163,    -1,   123,     1,    -1,
     103,     4,    -1,   103,    44,    -1,   103,     1,    -1,   106,
     163,    -1,   106,     1,    -1,   104,     4,     5,   163,    -1,
     104,    44,     5,   163,    -1,   104,     1,    -1,   105,     4,
       5,   163,    -1,   105,    44,     5,   163,    -1,   105,     1,
      -1,    79,    44,     5,     4,    -1,    79,    44,     5,    44,
      -1,    79,     1,    -1,   121,     4,    -1,   121,    44,    -1,
     121,     1,    -1,    76,    -1,    76,     1,    -1,    78,    -1,
      78,     1,    -1,    77,   163,    -1,    77,     1,    -1,   119,
      -1,   119,     1,    -1,   122,    44,    -1,   122,     1,    -1,
     144,    -1,   144,     1,    -1,   117,    44,     5,   165,     5,
     249,    -1,   117,     1,    -1,   118,    44,     5,   165,     5,
     249,    -1,   118,     1,    -1,    44,     5,     4,    -1,   249,
       5,    44,     5,     4,    -1,   125,    44,     5,   163,    -1,
     125,     4,     5,   163,    -1,   125,     1,    -1,   126,    41,
       5,    41,    -1,   126,    41,     5,    44,    -1,   126,    44,
       5,    41,    -1,   126,    44,     5,    44,    -1,   126,     1,
      -1,   127,    41,     5,    41,    -1,   127,    41,     5,    44,
      -1,   127,    44,     5,    41,    -1,   127,    44,     5,    44,
      -1,   127,     1,    -1,   128,    41,     5,    41,    -1,   128,
      41,     5,    44,    -1,   128,    44,     5,    41,    -1,   128,
      44,     5,    44,    -1,   128,     1,    -1,   129,    41,    -1,
     129,    44,    -1,   129,     1,    -1,   131,   163,     5,   163,
      -1,   131,     1,    -1,   132,   163,     5,   163,    -1,   132,
       1,    -1,   133,   165,     5,   164,    -1,   133,     1,    -1,
     134,   165,     5,   164,    -1,   134,     1,    -1,   135,   165,
       5,   164,    -1,   135,     1,    -1,   130,   165,     5,   164,
      -1,   130,     1,    -1,    69,   165,     5,   164,    -1,    69,
       1,    -1,   136,   165,    -1,   136,     1,    -1,   137,   165,
       5,   165,    -1,   137,   165,     5,    44,    -1,   137,     1,
      -1,   138,   165,     5,   165,    -1,   138,   165,     5,    44,
      -1,   138,     1,    -1,   139,   165,     5,   165,    -1,   139,
     165,     5,    44,    -1,   139,     1,    -1,   140,   165,     5,
     165,    -1,   140,   165,     5,    44,    -1,   140,     1,    -1,
     141,   163,     5,   164,    -1,   141,     1,    -1,   142,   163,
       5,   164,    -1,   142,     1,    -1,   143,   163,     5,   164,
      -1,   143,     1,    -1,   146,   165,    -1,   146,     1,    -1,
     147,   165,    -1,   147,     1,    -1,   148,   163,     5,   163,
      -1,   148,     1,    -1,   149,   163,     5,   163,    -1,   149,
       1,    -1,   150,   163,     5,   163,    -1,   150,     1,    -1,
     151,   163,     5,   163,    -1,   151,     1,    -1,   152,   163,
       5,   163,    -1,   152,     1,    -1,   153,   163,     5,   163,
      -1,   153,     1,    -1,   154,   163,     5,   163,    -1,   154,
       1,    -1,   155,   170,    -1,   155,   165,    -1,   155,     1,
      -1,   156,   170,    -1,   156,   165,    -1,   156,     1,    -1,
     157,   163,    -1,   157,     1,    -1,   158,   163,    -1,   158,
       1,    -1,   159,   165,     5,   163,    -1,   159,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   230,   230,   232,   236,   237,   238,   239,   240,   241,
     244,   244,   245,   245,   246,   246,   247,   247,   247,   247,
     247,   247,   249,   249,   250,   250,   250,   250,   251,   251,
     251,   252,   252,   253,   253,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   304,   305,   306,   307,   308,   313,   319,   328,   329,
     333,   334,   337,   339,   341,   342,   346,   347,   351,   352,
     356,   357,   358,   359,   360,   361,   362,   363,   364,   365,
     366,   367,   368,   369,   370,   371,   372,   373,   374,   375,
     376,   377,   378,   379,   380,   381,   382,   383,   384,   385,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     406,   407,   408,   409,   410,   411,   412,   413,   414,   415,
     416,   417,   418,   419,   420,   421,   422,   423,   424,   425,
     426,   427,   428,   429,   430,   431,   432,   433,   434,   435,
     436,   437,   438,   439,   440,   441,   442,   443,   444,   445,
     446,   447,   448,   449,   450,   451,   452,   453,   454,   455,
     456,   457,   461,   462,   466,   467,   471,   472,   476,   477,
     481,   482,   487,   488,   492,   493,   497,   498,   502,   503,
     508,   509,   513,   514,   518,   519,   523,   524,   529,   530,
     534,   535,   539,   540,   544,   545,   549,   550,   554,   555,
     559,   560,   561,   565,   566,   570,   571,   576,   577,   581,
     582,   587,   588,   592,   593,   597,   598,   599,   603,   604,
     605,   609,   610,   614,   615,   620,   621,   622,   626,   627,
     632,   633,   637,   638,   642,   643,   648,   649,   650,   654,
     655,   656,   660,   661,   662,   666,   667,   668,   672,   673,
     674,   678,   679,   680,   684,   685,   686,   690,   691,   692,
     696,   697,   698,   702,   703,   704,   708,   709,   710,   714,
     715,   716,   720,   721,   722,   726,   727,   731,   732,   736,
     737,   741,   742,   746,   747,   751,   752,   756,   757,   758,
     762,   763,   767,   768,   769,   773,   774,   775,   780,   781,
     782,   786,   787,   788,   792,   793,   797,   798,   802,   803,
     807,   808,   812,   813,   817,   818,   822,   823,   827,   828,
     832,   841,   850,   851,   852,   856,   857,   858,   859,   860,
     864,   865,   866,   867,   868,   872,   873,   874,   875,   876,
     880,   881,   882,   886,   887,   891,   892,   896,   897,   901,
     902,   906,   907,   911,   912,   916,   917,   921,   922,   926,
     927,   928,   932,   933,   934,   938,   939,   940,   944,   945,
     946,   951,   952,   956,   957,   961,   962,   966,   967,   971,
     972,   976,   977,   981,   982,   986,   987,   991,   992,   996,
     997,  1001,  1002,  1006,  1007,  1011,  1012,  1013,  1017,  1018,
    1019,  1023,  1024,  1028,  1029,  1034,  1035
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
  "I_TRAC", "I_WRT", "I_STO", "$accept", "input", "line", "xoperand",
  "operand", "op_variable", "op_register", "x_op_immediate",
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
  "inst_wrt", "inst_sto", 0
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
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   160,   161,   161,   162,   162,   162,   162,   162,   162,
     163,   163,   164,   164,   165,   165,   166,   166,   166,   166,
     166,   166,   167,   167,   168,   168,   168,   168,   169,   169,
     169,   170,   170,   171,   171,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     173,   172,   172,   172,   172,   172,   172,   172,   174,   174,
     175,   175,   176,   177,   177,   177,   178,   178,   179,   179,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   181,   181,   182,   182,   183,   183,   184,   184,
     185,   185,   186,   186,   187,   187,   188,   188,   189,   189,
     190,   190,   191,   191,   192,   192,   193,   193,   194,   194,
     195,   195,   196,   196,   197,   197,   198,   198,   199,   199,
     200,   200,   200,   201,   201,   202,   202,   203,   203,   204,
     204,   205,   205,   206,   206,   207,   207,   207,   208,   208,
     208,   209,   209,   210,   210,   211,   211,   211,   212,   212,
     213,   213,   214,   214,   215,   215,   216,   216,   216,   217,
     217,   217,   218,   218,   218,   219,   219,   219,   220,   220,
     220,   221,   221,   221,   222,   222,   222,   223,   223,   223,
     224,   224,   224,   225,   225,   225,   226,   226,   226,   227,
     227,   227,   228,   228,   228,   229,   229,   230,   230,   231,
     231,   232,   232,   233,   233,   234,   234,   235,   235,   235,
     236,   236,   237,   237,   237,   238,   238,   238,   239,   239,
     239,   240,   240,   240,   241,   241,   242,   242,   243,   243,
     244,   244,   245,   245,   246,   246,   247,   247,   248,   248,
     249,   249,   250,   250,   250,   251,   251,   251,   251,   251,
     252,   252,   252,   252,   252,   253,   253,   253,   253,   253,
     254,   254,   254,   255,   255,   256,   256,   257,   257,   258,
     258,   259,   259,   260,   260,   261,   261,   262,   262,   263,
     263,   263,   264,   264,   264,   265,   265,   265,   266,   266,
     266,   267,   267,   268,   268,   269,   269,   270,   270,   271,
     271,   272,   272,   273,   273,   274,   274,   275,   275,   276,
     276,   277,   277,   278,   278,   279,   279,   279,   280,   280,
     280,   281,   281,   282,   282,   283,   283
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     4,     4,
       5,     3,     4,     3,     4,     3,     3,     2,     3,     3,
       4,     1,     2,     3,     4,     5,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     6,     1,     3,     3,     3,
       2,     2,     4,     5,     3,     4,     2,     3,     2,     1,
       0,     4,     2,     3,     2,     2,     2,     2,     1,     3,
       1,     3,     2,     0,     1,     3,     1,     1,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     4,     2,     4,     2,     2,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     4,     4,     2,     4,     4,
       2,     4,     2,     4,     2,     2,     1,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     4,     4,     2,     6,
       6,     2,     6,     6,     2,     6,     6,     2,     4,     4,
       2,     4,     4,     2,     6,     6,     2,     6,     6,     2,
       6,     6,     2,     4,     4,     2,     6,     6,     2,     6,
       6,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     6,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     4,     4,     2,     4,     4,     2,     4,     4,
       2,     2,     2,     2,     1,     2,     1,     2,     2,     2,
       1,     2,     2,     2,     1,     2,     6,     2,     6,     2,
       3,     5,     4,     4,     2,     4,     4,     4,     4,     2,
       4,     4,     4,     4,     2,     4,     4,     4,     4,     2,
       2,     2,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     2,     2,     4,
       4,     2,     4,     4,     2,     4,     4,     2,     4,     4,
       2,     4,     2,     4,     2,     4,     2,     2,     2,     2,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    35,     0,     0,     0,     0,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
      79,     0,     0,     0,     0,     0,     0,     0,     0,    66,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   266,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     3,     0,     0,     0,   100,   165,   101,   102,
     108,   103,   109,   104,   110,   105,   111,   106,   107,   116,
     117,   118,   119,   120,   121,   159,   112,   113,   114,   115,
     122,   123,   153,   154,   151,   152,   124,   125,   126,   187,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   137,
     136,   138,   139,   140,   141,   142,   143,   144,   145,   162,
     146,   147,   148,   149,   150,   161,   155,   157,   156,   158,
     160,   166,   163,   164,   186,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   177,   178,   176,   179,   180,   181,
     182,   183,   184,   185,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,     9,    98,
       0,     0,    98,    98,    98,    47,    98,     0,     0,    98,
      76,    78,    82,    98,    36,    52,    98,    14,    16,    17,
      18,    19,    20,    21,     0,    15,     0,     0,     0,     0,
       0,     0,    84,    85,    86,    87,    88,    70,    90,    71,
      80,     0,    92,   203,     0,   207,   206,   209,    30,    24,
      29,    28,    25,    26,    10,     0,    11,    12,    13,    27,
     213,     0,   217,     0,   221,     0,   225,     0,   227,     0,
     211,     0,   215,     0,   219,     0,   223,     0,   396,     0,
     244,   243,   246,   245,   248,   247,   250,   249,   252,   251,
     254,   253,   345,   349,   348,   347,   340,     0,   267,   265,
     269,   268,   271,   270,   278,     0,   281,     0,   284,     0,
     287,     0,   290,     0,   293,     0,   296,     0,   299,     0,
     302,     0,   305,     0,   308,     0,     0,   311,     0,     0,
     314,   313,   312,   316,   315,   275,   274,   318,   317,   320,
     319,   322,     0,   324,   323,   329,   327,   328,   334,     0,
       0,   337,     0,     0,   331,   330,   229,     0,   231,     0,
     235,     0,   233,     0,   239,     0,   237,     0,   262,     0,
     264,     0,   257,     0,   260,     0,   357,     0,   359,     0,
     351,   242,   240,   241,   343,   341,   342,   353,   352,   326,
     325,   205,     0,   364,     0,     0,   369,     0,     0,   374,
       0,     0,   379,     0,     0,   382,   380,   381,   394,     0,
     384,     0,   386,     0,   388,     0,   390,     0,   392,     0,
     398,   397,   401,     0,   404,     0,   407,     0,   410,     0,
     412,     0,   414,     0,   416,     0,   355,   273,   272,   418,
     417,   420,   419,   422,     0,   424,     0,   426,     0,   428,
       0,   430,     0,   432,     0,   434,     0,   437,    32,    31,
     436,   435,   440,   439,   438,   442,   441,   444,   443,   446,
       0,     5,     7,     0,     8,    99,    37,    22,    98,    23,
      41,    43,    45,    46,    48,    49,    68,    67,    69,    74,
      77,    83,    53,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    93,    98,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
      38,    39,    42,    44,    50,    75,    34,    33,    54,    56,
      57,    58,    59,    64,    61,     0,    62,    63,    60,    89,
      91,    97,    96,    81,    94,    72,   202,   208,   212,   216,
     220,   224,   226,   210,   214,   218,   222,   395,   338,   339,
     276,   277,     0,     0,     0,     0,     0,     0,   288,   289,
     291,   292,     0,     0,     0,     0,     0,     0,   303,   304,
       0,     0,     0,     0,     0,   332,   333,   335,   336,   228,
     230,   234,   232,   238,   236,   261,   263,   256,   255,   259,
     258,     0,     0,   204,   363,   362,   365,   366,   367,   368,
     370,   371,   372,   373,   375,   376,   377,   378,   393,   383,
     385,   387,   389,   391,   400,   399,   403,   402,   406,   405,
     409,   408,   411,   413,   415,   421,   423,   425,   427,   429,
     431,   433,   445,    40,    55,     0,     0,    73,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    65,    95,   279,
     280,   282,   283,   285,   286,   294,   295,   297,   298,   300,
     301,   307,   306,   309,   310,   321,     0,   356,   358,     0,
       0,   360,     0,     0,   361
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   142,   305,   306,   307,   275,   652,   308,   309,
     511,   638,   143,   553,   287,   289,   144,   653,   654,   526,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   797,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -247
static const yytype_int16 yypact[] =
{
    -247,   778,  -247,     3,  -247,  -247,    33,    64,    95,   154,
     155,   167,   170,  -247,   179,   215,   234,   236,   237,   196,
    -247,   250,   271,   289,   290,   291,  1038,  1038,   438,  -247,
     230,   240,   244,   247,   259,   267,   268,   285,   304,    97,
     256,    82,   159,   177,   198,   214,   232,   457,   533,   628,
     725,  1344,  1358,  1368,  1385,  1395,   270,   288,   119,   467,
     169,     6,   551,   654,  -247,  1409,  1419,  1436,  1446,  1460,
    1470,  1487,  1497,  1511,  1521,  1538,    16,   176,   287,    17,
    1548,    65,    66,  1303,  1329,   455,   456,   465,   669,   710,
     937,   952,   967,   993,  1008,  1562,    70,   147,   172,   269,
     491,   187,   479,   480,   492,  1023,  1572,   485,    56,    72,
     138,   140,  1589,  1049,  1064,  1599,  1613,  1623,  1640,  1650,
    1664,  1674,  1691,  1079,  1105,  1120,   233,  1701,  1715,  1725,
    1135,  1161,  1176,  1191,  1217,  1232,  1247,    29,   116,  1273,
    1288,  1742,  -247,   309,   295,   327,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,   300,
     896,   896,   300,   300,   300,   303,   300,   420,   306,   300,
     307,  -247,  -247,   300,  -247,  -247,   300,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,   343,  -247,   345,   450,    58,   458,
     460,   462,  -247,  -247,  -247,  -247,  -247,   471,  -247,   472,
    -247,   466,  -247,  -247,   482,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,   483,  -247,  -247,  -247,  -247,
    -247,   489,  -247,   490,  -247,   521,  -247,   535,  -247,   536,
    -247,   549,  -247,   550,  -247,   558,  -247,   561,  -247,   568,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,   570,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,   571,  -247,   572,  -247,   579,
    -247,   580,  -247,   581,  -247,   582,  -247,   603,  -247,   623,
    -247,   626,  -247,   627,  -247,   637,   638,  -247,   639,   651,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,   652,  -247,  -247,  -247,  -247,  -247,  -247,   653,
     655,  -247,   659,   660,  -247,  -247,  -247,   661,  -247,   662,
    -247,   663,  -247,   674,  -247,   675,  -247,   676,  -247,   677,
    -247,   686,  -247,   687,  -247,   688,  -247,   689,  -247,   691,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,   692,  -247,   720,   723,  -247,   724,   726,  -247,
     729,   730,  -247,   731,   732,  -247,  -247,  -247,  -247,   735,
    -247,   747,  -247,   748,  -247,   762,  -247,   764,  -247,   771,
    -247,  -247,  -247,   772,  -247,   775,  -247,   777,  -247,   779,
    -247,   790,  -247,   814,  -247,   815,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,   821,  -247,   822,  -247,   825,  -247,
     826,  -247,   831,  -247,   934,  -247,   936,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
     938,  -247,  -247,   459,  -247,  -247,   552,  -247,   300,  -247,
     591,   617,  -247,  -247,  -247,   741,  -247,  -247,  -247,   900,
    -247,  -247,     4,    15,    28,   493,   940,   901,   942,   973,
     975,   939,   992,  1763,   300,   911,   911,   911,   911,   911,
     911,   911,   911,   911,   911,   911,  1778,   145,   512,   512,
     512,   512,   512,   512,   512,   512,   512,   512,   911,   911,
    1031,  1045,  1778,   911,   911,   911,   911,   911,   911,   911,
     911,   911,   911,  1038,  1038,   589,  1753,  1038,  1038,   911,
     911,   911,   -29,   -10,    -8,    -6,    43,    44,  1778,   911,
     911,  1778,  1778,  1778,   566,   700,   926,   982,  1778,  1778,
    1778,   911,   911,   911,   911,   911,   911,   911,   911,  -247,
    -247,   994,  -247,  -247,  -247,  -247,  -247,  -247,     4,  -247,
    -247,  -247,  -247,  -247,  -247,  1077,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  1086,  -247,  1050,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  1130,  1133,  1134,  1136,  1137,  1138,  -247,  -247,
    -247,  -247,  1139,  1140,  1142,  1143,  1186,  1189,  -247,  -247,
    1190,  1192,  1193,  1194,  1195,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  1196,  1198,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  1200,  1763,  -247,  1038,  1038,
     911,   911,   911,   911,  1038,  1038,   911,   911,  1038,  1038,
     911,   911,  1096,  1152,   896,  1203,  1203,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  1245,  1246,  1246,  1248,
    1209,  -247,  1249,  1251,  -247
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -247,  -247,  -247,    62,   -81,   -26,  -247,  -241,  -246,  1266,
     592,   618,  -247,  -247,  -247,  -247,  -247,  -247,   501,  -201,
    1115,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,   484,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -355
static const yytype_int16 yytable[] =
{
     274,   276,   392,   394,   529,   529,   248,   346,   636,   528,
     530,   529,   716,   294,   296,   717,   537,   374,   383,   639,
     375,   321,   323,   325,   327,   329,   331,   333,   335,   337,
     507,   718,   641,   720,   719,   722,   721,   249,   723,   353,
     355,   357,   359,   361,   363,   365,   367,   369,   371,   373,
     347,   531,   532,   533,   386,   535,   637,   446,   539,   640,
     376,   384,   541,   546,   547,   542,   387,   389,   250,   419,
     267,   420,   642,   449,   268,   269,   270,   271,   272,   273,
     442,   508,   509,   297,   724,   726,   459,   725,   727,   465,
     467,   469,   471,   473,   475,   477,   479,   447,   293,   251,
     448,   488,   490,   492,   311,   313,   315,   317,   319,   388,
     390,   510,   513,   450,   421,   520,   451,   512,   339,   341,
     342,   344,  -344,   267,   349,   351,   298,   268,   269,   270,
     271,   272,   273,   299,   300,   301,   302,   303,   267,   452,
     304,   455,   268,   269,   270,   271,   272,   273,   422,   668,
     405,   407,   409,   411,   413,   415,   417,   267,   252,   253,
     310,   268,   269,   270,   271,   272,   273,   440,   508,   509,
     345,   254,  -346,   424,   255,   461,   463,   377,   312,   453,
     378,   456,   454,   256,   457,   481,   483,   485,   430,   669,
    -350,   423,   494,   496,   498,   500,   502,   504,   506,   314,
     267,   516,   518,   298,   268,   269,   270,   271,   272,   273,
     299,   300,   301,   302,   303,   316,   425,   304,   267,   257,
     379,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,   318,   486,   304,  -354,   261,   258,   267,
     259,   260,   298,   268,   269,   270,   271,   272,   273,   299,
     300,   301,   302,   303,   262,   267,   304,   295,   298,   268,
     269,   270,   271,   272,   273,   299,   300,   301,   302,   303,
     426,   338,   304,   267,   282,   263,   298,   268,   269,   270,
     271,   272,   273,   299,   300,   301,   302,   303,   380,   340,
     304,   381,   283,   264,   265,   266,   284,   267,   522,   285,
     286,   268,   269,   270,   271,   272,   273,   529,   288,   290,
     292,   267,   521,   427,   298,   268,   269,   270,   271,   272,
     273,   299,   300,   301,   302,   303,   291,   631,   304,   267,
     524,   382,   298,   268,   269,   270,   271,   272,   273,   299,
     300,   301,   302,   303,   525,   534,   304,   538,   543,   540,
     544,    39,    40,   655,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   545,   395,   398,   320,   396,
     399,   536,   629,   548,   298,   549,   401,   550,   343,   402,
     554,   299,   300,   301,   302,   303,   551,   552,   527,   277,
     431,   434,   278,   432,   435,   667,   443,   555,   556,   444,
     279,   280,   428,   437,   557,   558,   281,   643,   267,   397,
     400,   694,   268,   269,   270,   271,   272,   273,   267,   403,
     529,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,   433,   436,   304,   559,   728,   529,   445,
     731,   732,   733,   795,   322,   429,   438,   742,   743,   744,
     560,   561,   670,   672,   674,   676,   678,   680,   682,   684,
     686,   688,   348,   267,   562,   563,   298,   268,   269,   270,
     271,   272,   273,   564,   300,   301,   565,   705,   706,   708,
     710,   711,   712,   566,   267,   567,   568,   569,   268,   269,
     270,   271,   272,   273,   570,   571,   572,   573,   735,   737,
     739,   741,   267,   707,   630,   298,   268,   269,   270,   271,
     272,   273,   299,   300,   301,   302,   303,   267,   574,   304,
     734,   268,   269,   270,   271,   272,   273,   656,   657,   658,
     659,   660,   661,   662,   663,   664,   665,   666,   575,   324,
     267,   576,   577,   632,   268,   269,   270,   271,   272,   273,
     690,   691,   578,   579,   580,   695,   696,   697,   698,   699,
     700,   701,   702,   703,   704,   350,   581,   582,   583,   633,
     584,   713,   714,   715,   585,   586,   587,   588,   589,   267,
     404,   729,   730,   268,   269,   270,   271,   272,   273,   590,
     591,   592,   593,   745,   746,   747,   748,   749,   750,   751,
     752,   594,   595,   596,   597,   267,   598,   599,   298,   268,
     269,   270,   271,   272,   273,   299,   300,   301,   302,   303,
     267,   406,   304,   298,   268,   269,   270,   271,   272,   273,
     299,   300,   301,   302,   303,   600,   326,   304,   601,   602,
     514,   603,   779,   780,   604,   605,   606,   607,   785,   786,
     608,   267,   789,   790,   736,   268,   269,   270,   271,   272,
     273,   267,   609,   610,   298,   268,   269,   270,   271,   272,
     273,   299,   300,   301,   302,   303,   267,   611,   304,   612,
     268,   269,   270,   271,   272,   273,   613,   614,     2,     3,
     615,     4,   616,   634,   617,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,   618,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,   619,
     620,    38,   781,   782,   783,   784,   621,   622,   787,   788,
     623,   624,   791,   792,    39,    40,   625,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   408,   626,
     298,   627,   635,   628,   644,   645,   646,   299,   300,   301,
     302,   303,   267,   410,   527,   298,   268,   269,   270,   271,
     272,   273,   299,   300,   301,   302,   303,   267,   412,   304,
     738,   268,   269,   270,   271,   272,   273,   647,   267,   648,
     649,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,   267,   414,   304,   298,   268,   269,   270,
     271,   272,   273,   299,   300,   301,   302,   303,   267,   416,
     304,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,   267,   439,   304,   740,   268,   269,   270,
     271,   272,   273,   650,   267,   692,   753,   298,   268,   269,
     270,   271,   272,   273,   299,   300,   301,   302,   303,   267,
     460,   304,   298,   268,   269,   270,   271,   272,   273,   299,
     300,   301,   302,   303,   267,   462,   304,   298,   268,   269,
     270,   271,   272,   273,   299,   300,   301,   302,   303,   267,
     480,   304,   755,   268,   269,   270,   271,   272,   273,   693,
     267,   756,   757,   298,   268,   269,   270,   271,   272,   273,
     299,   300,   301,   302,   303,   267,   482,   304,   298,   268,
     269,   270,   271,   272,   273,   299,   300,   301,   302,   303,
     267,   484,   304,   298,   268,   269,   270,   271,   272,   273,
     299,   300,   301,   302,   303,   758,   493,   304,   759,   760,
     793,   761,   762,   763,   764,   765,   267,   766,   767,   298,
     268,   269,   270,   271,   272,   273,   299,   300,   301,   302,
     303,   267,   495,   304,   298,   268,   269,   270,   271,   272,
     273,   299,   300,   301,   302,   303,   267,   497,   304,   298,
     268,   269,   270,   271,   272,   273,   299,   300,   301,   302,
     303,   768,   499,   304,   769,   770,   794,   771,   772,   773,
     774,   775,   267,   776,   777,   298,   268,   269,   270,   271,
     272,   273,   299,   300,   301,   302,   303,   267,   501,   304,
     298,   268,   269,   270,   271,   272,   273,   299,   300,   301,
     302,   303,   267,   503,   304,   298,   268,   269,   270,   271,
     272,   273,   299,   300,   301,   302,   303,   796,   505,   304,
     799,   800,   801,   802,   803,   804,   754,   778,   267,   523,
     798,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,   267,   515,   304,   298,   268,   269,   270,
     271,   272,   273,   299,   300,   301,   302,   303,   267,   517,
     304,   298,   268,   269,   270,   271,   272,   273,   299,   300,
     301,   302,   303,     0,   391,   304,     0,     0,     0,     0,
       0,     0,     0,     0,   267,     0,     0,   298,   268,   269,
     270,   271,   272,   273,   299,   300,   301,   302,   303,   267,
     393,   304,   298,   268,   269,   270,   271,   272,   273,   299,
     300,   301,   302,   303,   267,   328,   304,   298,   268,   269,
     270,   271,   272,   273,   299,   300,   301,   302,   303,   330,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   332,
     267,     0,     0,   298,   268,   269,   270,   271,   272,   273,
     299,   300,   301,   302,   303,   267,   334,     0,     0,   268,
     269,   270,   271,   272,   273,     0,   336,     0,     0,   267,
       0,     0,     0,   268,   269,   270,   271,   272,   273,   267,
     352,     0,     0,   268,   269,   270,   271,   272,   273,     0,
     354,     0,     0,     0,     0,     0,   267,     0,     0,     0,
     268,   269,   270,   271,   272,   273,   267,   356,     0,     0,
     268,   269,   270,   271,   272,   273,     0,   358,     0,     0,
     267,     0,     0,     0,   268,   269,   270,   271,   272,   273,
     267,   360,     0,     0,   268,   269,   270,   271,   272,   273,
       0,   362,     0,     0,     0,     0,     0,   267,     0,     0,
       0,   268,   269,   270,   271,   272,   273,   267,   364,     0,
       0,   268,   269,   270,   271,   272,   273,     0,   366,     0,
       0,   267,     0,     0,     0,   268,   269,   270,   271,   272,
     273,   267,   368,     0,     0,   268,   269,   270,   271,   272,
     273,     0,   370,     0,     0,     0,     0,     0,   267,     0,
       0,     0,   268,   269,   270,   271,   272,   273,   267,   372,
       0,     0,   268,   269,   270,   271,   272,   273,     0,   385,
       0,     0,   267,     0,     0,     0,   268,   269,   270,   271,
     272,   273,   267,   418,     0,     0,   268,   269,   270,   271,
     272,   273,     0,   441,     0,     0,     0,     0,     0,   267,
       0,     0,     0,   268,   269,   270,   271,   272,   273,   267,
     458,     0,     0,   268,   269,   270,   271,   272,   273,     0,
     464,     0,     0,   267,     0,     0,     0,   268,   269,   270,
     271,   272,   273,   267,   466,     0,     0,   268,   269,   270,
     271,   272,   273,     0,   468,     0,     0,     0,     0,     0,
     267,     0,     0,     0,   268,   269,   270,   271,   272,   273,
     267,   470,     0,     0,   268,   269,   270,   271,   272,   273,
       0,   472,     0,     0,   267,     0,     0,     0,   268,   269,
     270,   271,   272,   273,   267,   474,     0,     0,   268,   269,
     270,   271,   272,   273,     0,   476,     0,     0,     0,     0,
       0,   267,     0,     0,     0,   268,   269,   270,   271,   272,
     273,   267,   478,     0,     0,   268,   269,   270,   271,   272,
     273,     0,   487,     0,     0,   267,     0,     0,     0,   268,
     269,   270,   271,   272,   273,   267,   489,     0,     0,   268,
     269,   270,   271,   272,   273,     0,   491,     0,     0,     0,
       0,     0,   267,     0,     0,     0,   268,   269,   270,   271,
     272,   273,   267,   519,     0,     0,   268,   269,   270,   271,
     272,   273,     0,     0,     0,     0,   267,   709,     0,     0,
     268,   269,   270,   271,   272,   273,   267,     0,     0,     0,
     268,   269,   270,   271,   272,   273,     0,     0,     0,     0,
       0,     0,     0,   267,     0,     0,     0,   268,   269,   270,
     271,   272,   273,     0,   267,     0,     0,     0,   268,   269,
     270,   271,   272,   273,   651,     0,     0,   298,     0,     0,
       0,     0,     0,     0,   299,   300,   301,   302,   303,   267,
       0,   527,   298,   268,   269,   270,   271,   272,   273,   299,
     300,   301,   302,   303,   671,   673,   675,   677,   679,   681,
     683,   685,   687,   689
};

static const yytype_int16 yycheck[] =
{
      26,    27,    83,    84,   250,   251,     3,     1,     4,   250,
     251,   257,    41,    39,    40,    44,   257,     1,     1,     4,
       4,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       1,    41,     4,    41,    44,    41,    44,     4,    44,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      44,   252,   253,   254,    80,   256,    52,     1,   259,    44,
      44,    44,   263,     5,     6,   266,     1,     1,     4,    95,
      41,     1,    44,     1,    45,    46,    47,    48,    49,    50,
     106,    52,    53,     1,    41,    41,   112,    44,    44,   115,
     116,   117,   118,   119,   120,   121,   122,    41,     1,     4,
      44,   127,   128,   129,    42,    43,    44,    45,    46,    44,
      44,   137,   138,    41,    44,   141,    44,     1,    56,    57,
       1,    59,     3,    41,    62,    63,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,     1,    45,    46,    47,    48,    49,    50,     1,     4,
      88,    89,    90,    91,    92,    93,    94,    41,     4,     4,
       1,    45,    46,    47,    48,    49,    50,   105,    52,    53,
       1,     4,     3,     1,     4,   113,   114,     1,     1,    41,
       4,    41,    44,     4,    44,   123,   124,   125,     1,    44,
       3,    44,   130,   131,   132,   133,   134,   135,   136,     1,
      41,   139,   140,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     1,    44,    58,    41,     4,
      44,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     1,     1,    58,     3,    41,     4,    41,
       4,     4,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,     4,    41,    58,     1,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       1,     1,    58,    41,    44,     4,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,     1,     1,
      58,     4,    52,     4,     4,     4,    52,    41,     3,    52,
      41,    45,    46,    47,    48,    49,    50,   553,    41,    41,
       6,    41,     3,    44,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    41,   528,    58,    41,
       3,    44,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    44,    42,    58,    41,     5,    42,
       5,    56,    57,   554,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,     5,     1,     1,     1,     4,
       4,    41,     3,     5,    44,     5,     1,     5,     1,     4,
       4,    51,    52,    53,    54,    55,     5,     5,    58,    41,
       1,     1,    44,     4,     4,   566,     1,     5,     5,     4,
      52,    53,     1,     1,     5,     5,    58,     4,    41,    44,
      44,   582,    45,    46,    47,    48,    49,    50,    41,    44,
     756,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    44,    44,    58,     5,   608,   774,    44,
     611,   612,   613,   774,     1,    44,    44,   618,   619,   620,
       5,     5,   568,   569,   570,   571,   572,   573,   574,   575,
     576,   577,     1,    41,     5,     5,    44,    45,    46,    47,
      48,    49,    50,     5,    52,    53,     5,   593,   594,   595,
     596,   597,   598,     5,    41,     5,     5,     5,    45,    46,
      47,    48,    49,    50,     5,     5,     5,     5,   614,   615,
     616,   617,    41,     4,    42,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     5,    58,
      44,    45,    46,    47,    48,    49,    50,   555,   556,   557,
     558,   559,   560,   561,   562,   563,   564,   565,     5,     1,
      41,     5,     5,    42,    45,    46,    47,    48,    49,    50,
     578,   579,     5,     5,     5,   583,   584,   585,   586,   587,
     588,   589,   590,   591,   592,     1,     5,     5,     5,    42,
       5,   599,   600,   601,     5,     5,     5,     5,     5,    41,
       1,   609,   610,    45,    46,    47,    48,    49,    50,     5,
       5,     5,     5,   621,   622,   623,   624,   625,   626,   627,
     628,     5,     5,     5,     5,    41,     5,     5,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,     1,    58,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     5,     1,    58,     5,     5,
     138,     5,   758,   759,     5,     5,     5,     5,   764,   765,
       5,    41,   768,   769,    44,    45,    46,    47,    48,    49,
      50,    41,     5,     5,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    41,     5,    58,     5,
      45,    46,    47,    48,    49,    50,     5,     5,     0,     1,
       5,     3,     5,    42,     5,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,     5,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,     5,
       5,    43,   760,   761,   762,   763,     5,     5,   766,   767,
       5,     5,   770,   771,    56,    57,     5,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,     1,     5,
      44,     5,    42,     5,     4,    44,     4,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,     4,    41,     4,
      41,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    41,    41,     4,    42,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,     5,    45,    46,    47,    48,    49,    50,    44,
      41,     5,    42,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    58,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,     1,    58,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     5,     1,    58,     5,     5,
      44,     5,     5,     5,     5,     5,    41,     5,     5,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    41,     1,    58,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    41,     1,    58,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,     5,     1,    58,     5,     5,    44,     5,     5,     5,
       5,     5,    41,     5,     4,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    44,     1,    58,
       5,     5,     4,    44,     5,     4,   638,   756,    41,   144,
     776,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    -1,     1,    58,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      41,    -1,    -1,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    -1,     1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,    41,
       1,    -1,    -1,    45,    46,    47,    48,    49,    50,    -1,
       1,    -1,    -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    41,     1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    -1,     1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      41,     1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    41,    -1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    41,     1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    -1,     1,    -1,
      -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    41,     1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    -1,     1,    -1,    -1,    -1,    -1,    -1,    41,    -1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    41,     1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    -1,     1,
      -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    41,     1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    -1,     1,    -1,    -1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,    41,
       1,    -1,    -1,    45,    46,    47,    48,    49,    50,    -1,
       1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    41,     1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    -1,     1,    -1,    -1,    -1,    -1,    -1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      41,     1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      -1,     1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    41,     1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    41,     1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    -1,     1,    -1,    -1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    41,     1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    -1,     1,    -1,    -1,    -1,
      -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    41,     1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    -1,    -1,    -1,    -1,    41,     4,    -1,    -1,
      45,    46,    47,    48,    49,    50,    41,    -1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    41,    -1,    -1,    44,    -1,    -1,
      -1,    -1,    -1,    -1,    51,    52,    53,    54,    55,    41,
      -1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,   568,   569,   570,   571,   572,   573,
     574,   575,   576,   577
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   161,     0,     1,     3,     7,     8,     9,    10,    11,
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
     158,   159,   162,   172,   176,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,     3,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,    41,     4,     4,     4,     4,     4,    41,    45,    46,
      47,    48,    49,    50,   165,   166,   165,    41,    44,    52,
      53,    58,    44,    52,    52,    52,    41,   174,    41,   175,
      41,    41,     6,     1,   165,     1,   165,     1,    44,    51,
      52,    53,    54,    55,    58,   163,   164,   165,   168,   169,
       1,   163,     1,   163,     1,   163,     1,   163,     1,   163,
       1,   165,     1,   165,     1,   165,     1,   165,     1,   165,
       1,   165,     1,   165,     1,   165,     1,   165,     1,   163,
       1,   163,     1,     1,   163,     1,     1,    44,     1,   163,
       1,   163,     1,   165,     1,   165,     1,   165,     1,   165,
       1,   165,     1,   165,     1,   165,     1,   165,     1,   165,
       1,   165,     1,   165,     1,     4,    44,     1,     4,    44,
       1,     4,    44,     1,    44,     1,   165,     1,    44,     1,
      44,     1,   164,     1,   164,     1,     4,    44,     1,     4,
      44,     1,     4,    44,     1,   163,     1,   163,     1,   163,
       1,   163,     1,   163,     1,   163,     1,   163,     1,   165,
       1,    44,     1,    44,     1,    44,     1,    44,     1,    44,
       1,     1,     4,    44,     1,     4,    44,     1,    44,     1,
     163,     1,   165,     1,     4,    44,     1,    41,    44,     1,
      41,    44,     1,    41,    44,     1,    41,    44,     1,   165,
       1,   163,     1,   163,     1,   165,     1,   165,     1,   165,
       1,   165,     1,   165,     1,   165,     1,   165,     1,   165,
       1,   163,     1,   163,     1,   163,     1,     1,   165,     1,
     165,     1,   165,     1,   163,     1,   163,     1,   163,     1,
     163,     1,   163,     1,   163,     1,   163,     1,    52,    53,
     165,   170,     1,   165,   170,     1,   163,     1,   163,     1,
     165,     3,     3,   180,     3,    44,   179,    58,   167,   168,
     167,   179,   179,   179,    42,   179,    41,   167,    41,   179,
      42,   179,   179,     5,     5,     5,     5,     6,     5,     5,
       5,     5,     5,   173,     4,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     3,
      42,   179,    42,    42,    42,    42,     4,    52,   171,     4,
      44,     4,    44,     4,     4,    44,     4,     4,     4,    41,
      41,    41,   167,   177,   178,   179,   163,   163,   163,   163,
     163,   163,   163,   163,   163,   163,   163,   164,     4,    44,
     165,   169,   165,   169,   165,   169,   165,   169,   165,   169,
     165,   169,   165,   169,   165,   169,   165,   169,   165,   169,
     163,   163,     4,    44,   164,   163,   163,   163,   163,   163,
     163,   163,   163,   163,   163,   165,   165,     4,   165,     4,
     165,   165,   165,   163,   163,   163,    41,    44,    41,    44,
      41,    44,    41,    44,    41,    44,    41,    44,   164,   163,
     163,   164,   164,   164,    44,   165,    44,   165,    44,   165,
      44,   165,   164,   164,   164,   163,   163,   163,   163,   163,
     163,   163,   163,    42,   171,     5,     5,    42,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     4,   178,   165,
     165,   163,   163,   163,   163,   165,   165,   163,   163,   165,
     165,   163,   163,    44,    44,   167,    44,   249,   249,     5,
       5,     4,    44,     5,     4
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
# if YYLTYPE_IS_TRIVIAL
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
#line 241 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 35:
#line 257 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 36:
#line 258 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 37:
#line 259 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 38:
#line 260 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 39:
#line 261 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 40:
#line 262 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 41:
#line 263 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 264 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 265 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 266 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 45:
#line 267 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 46:
#line 268 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 47:
#line 269 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 48:
#line 270 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 49:
#line 271 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 50:
#line 272 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 51:
#line 273 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 52:
#line 274 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]) ); }
    break;

  case 53:
#line 275 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 54:
#line 276 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 55:
#line 277 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]) );
   }
    break;

  case 56:
#line 280 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 57:
#line 281 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 58:
#line 282 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 59:
#line 283 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 60:
#line 284 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 61:
#line 285 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 62:
#line 286 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 63:
#line 287 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 64:
#line 288 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 65:
#line 289 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 66:
#line 290 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 67:
#line 291 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 68:
#line 292 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 69:
#line 293 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 72:
#line 296 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 73:
#line 297 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 74:
#line 298 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 75:
#line 299 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 76:
#line 300 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 77:
#line 301 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 78:
#line 302 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 79:
#line 303 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 80:
#line 304 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 82:
#line 305 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 83:
#line 306 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 84:
#line 307 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 85:
#line 309 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 86:
#line 314 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 87:
#line 320 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 88:
#line 328 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 89:
#line 329 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 90:
#line 333 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 91:
#line 334 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 92:
#line 337 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 94:
#line 341 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); }
    break;

  case 95:
#line 342 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); }
    break;

  case 98:
#line 351 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 202:
#line 461 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 203:
#line 462 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 204:
#line 466 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 205:
#line 467 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 206:
#line 471 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 207:
#line 472 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 208:
#line 476 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 209:
#line 477 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 210:
#line 481 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 211:
#line 482 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 212:
#line 487 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 213:
#line 488 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 214:
#line 492 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 215:
#line 493 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 216:
#line 497 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 217:
#line 498 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 218:
#line 502 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 219:
#line 503 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 220:
#line 508 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 221:
#line 509 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 222:
#line 513 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 223:
#line 514 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 224:
#line 518 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 225:
#line 519 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 226:
#line 523 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 227:
#line 524 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 228:
#line 529 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 229:
#line 530 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 230:
#line 534 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 231:
#line 535 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 232:
#line 539 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 233:
#line 540 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 234:
#line 544 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 235:
#line 545 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 236:
#line 549 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 237:
#line 550 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 238:
#line 554 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 239:
#line 555 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 240:
#line 559 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 241:
#line 560 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 242:
#line 561 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 243:
#line 565 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 244:
#line 566 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 245:
#line 570 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 246:
#line 571 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 247:
#line 576 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 248:
#line 577 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 249:
#line 581 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 250:
#line 582 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 251:
#line 587 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 252:
#line 588 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 253:
#line 592 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 254:
#line 593 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 255:
#line 597 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 256:
#line 598 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 257:
#line 599 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 258:
#line 603 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 259:
#line 604 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 260:
#line 605 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 261:
#line 609 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 262:
#line 610 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 263:
#line 614 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 264:
#line 615 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 265:
#line 620 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 266:
#line 621 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 267:
#line 622 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 268:
#line 626 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 269:
#line 627 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 270:
#line 632 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 271:
#line 633 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 272:
#line 637 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 273:
#line 638 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 274:
#line 642 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 275:
#line 643 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 276:
#line 648 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 277:
#line 649 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 278:
#line 650 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 279:
#line 654 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 280:
#line 655 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 281:
#line 656 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 282:
#line 660 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 283:
#line 661 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 284:
#line 662 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 285:
#line 666 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 286:
#line 667 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 287:
#line 668 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 288:
#line 672 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 289:
#line 673 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 290:
#line 674 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 291:
#line 678 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 292:
#line 679 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 293:
#line 680 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 294:
#line 684 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 295:
#line 685 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 296:
#line 686 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 297:
#line 690 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 298:
#line 691 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 299:
#line 692 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 300:
#line 696 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 301:
#line 697 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 302:
#line 698 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 303:
#line 702 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 304:
#line 703 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 305:
#line 704 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 306:
#line 708 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 307:
#line 709 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 308:
#line 710 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 309:
#line 714 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 310:
#line 715 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 311:
#line 716 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 312:
#line 720 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 313:
#line 721 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 314:
#line 722 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 315:
#line 726 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 316:
#line 727 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 317:
#line 731 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 318:
#line 732 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 319:
#line 736 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 320:
#line 737 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 321:
#line 741 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 322:
#line 742 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 323:
#line 746 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 324:
#line 747 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 325:
#line 751 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 326:
#line 752 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 327:
#line 756 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 328:
#line 757 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 329:
#line 758 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 330:
#line 762 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 331:
#line 763 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 332:
#line 767 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 333:
#line 768 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 334:
#line 769 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 335:
#line 773 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 336:
#line 774 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 337:
#line 775 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 338:
#line 780 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 339:
#line 781 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 340:
#line 782 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 341:
#line 786 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 342:
#line 787 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 343:
#line 788 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 344:
#line 792 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 345:
#line 793 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 346:
#line 797 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 347:
#line 798 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 348:
#line 802 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 349:
#line 803 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 350:
#line 807 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 351:
#line 808 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 352:
#line 812 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 353:
#line 813 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 354:
#line 817 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 355:
#line 818 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 356:
#line 822 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 357:
#line 823 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 358:
#line 827 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 359:
#line 828 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 360:
#line 833 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 361:
#line 842 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 362:
#line 850 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 363:
#line 851 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 364:
#line 852 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 365:
#line 856 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 366:
#line 857 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 367:
#line 858 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 368:
#line 859 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 369:
#line 860 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 370:
#line 864 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 371:
#line 865 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 372:
#line 866 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 373:
#line 867 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 374:
#line 868 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 375:
#line 872 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 376:
#line 873 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 377:
#line 874 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 378:
#line 875 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 379:
#line 876 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 380:
#line 880 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 381:
#line 881 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 382:
#line 882 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 383:
#line 886 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 384:
#line 887 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 385:
#line 891 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 386:
#line 892 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 387:
#line 896 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 388:
#line 897 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 389:
#line 901 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 390:
#line 902 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 391:
#line 906 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 392:
#line 907 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 393:
#line 911 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 394:
#line 912 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 395:
#line 916 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 396:
#line 917 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 397:
#line 921 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 398:
#line 922 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 399:
#line 926 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 400:
#line 927 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 401:
#line 928 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 402:
#line 932 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 403:
#line 933 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 404:
#line 934 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 405:
#line 938 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 406:
#line 939 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 407:
#line 940 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 408:
#line 944 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 409:
#line 945 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 410:
#line 946 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 411:
#line 951 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 412:
#line 952 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 413:
#line 956 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 414:
#line 957 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 415:
#line 961 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 416:
#line 962 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 417:
#line 966 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 418:
#line 967 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 419:
#line 971 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 420:
#line 972 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 421:
#line 976 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 422:
#line 977 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 423:
#line 981 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 424:
#line 982 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 425:
#line 986 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 426:
#line 987 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 427:
#line 991 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 428:
#line 992 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 429:
#line 996 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 430:
#line 997 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 431:
#line 1001 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 432:
#line 1002 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 433:
#line 1006 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 434:
#line 1007 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 435:
#line 1011 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 436:
#line 1012 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 437:
#line 1013 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 438:
#line 1017 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 439:
#line 1018 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 440:
#line 1019 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 441:
#line 1023 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 442:
#line 1024 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 443:
#line 1028 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 444:
#line 1029 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 445:
#line 1034 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 446:
#line 1035 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4224 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"
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


#line 1039 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


