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
     DSWITCH = 283,
     DSELECT = 284,
     DCASE = 285,
     DENDSWITCH = 286,
     DLINE = 287,
     DSTRING = 288,
     DCSTRING = 289,
     DHAS = 290,
     DHASNT = 291,
     DINHERIT = 292,
     DINSTANCE = 293,
     SYMBOL = 294,
     EXPORT = 295,
     LABEL = 296,
     INTEGER = 297,
     REG_A = 298,
     REG_B = 299,
     REG_S1 = 300,
     REG_S2 = 301,
     NUMERIC = 302,
     STRING = 303,
     STRING_ID = 304,
     TRUE_TOKEN = 305,
     FALSE_TOKEN = 306,
     I_LD = 307,
     I_LNIL = 308,
     NIL = 309,
     I_ADD = 310,
     I_SUB = 311,
     I_MUL = 312,
     I_DIV = 313,
     I_MOD = 314,
     I_POW = 315,
     I_ADDS = 316,
     I_SUBS = 317,
     I_MULS = 318,
     I_DIVS = 319,
     I_POWS = 320,
     I_INC = 321,
     I_DEC = 322,
     I_NEG = 323,
     I_NOT = 324,
     I_RET = 325,
     I_RETV = 326,
     I_RETA = 327,
     I_FORK = 328,
     I_PUSH = 329,
     I_PSHR = 330,
     I_PSHN = 331,
     I_POP = 332,
     I_LDV = 333,
     I_LDVT = 334,
     I_STV = 335,
     I_STVR = 336,
     I_STVS = 337,
     I_LDP = 338,
     I_LDPT = 339,
     I_STP = 340,
     I_STPR = 341,
     I_STPS = 342,
     I_TRAV = 343,
     I_TRAN = 344,
     I_TRAL = 345,
     I_IPOP = 346,
     I_XPOP = 347,
     I_GENA = 348,
     I_GEND = 349,
     I_GENR = 350,
     I_GEOR = 351,
     I_JMP = 352,
     I_IFT = 353,
     I_IFF = 354,
     I_BOOL = 355,
     I_EQ = 356,
     I_NEQ = 357,
     I_GT = 358,
     I_GE = 359,
     I_LT = 360,
     I_LE = 361,
     I_UNPK = 362,
     I_UNPS = 363,
     I_CALL = 364,
     I_INST = 365,
     I_SWCH = 366,
     I_SELE = 367,
     I_NOP = 368,
     I_TRY = 369,
     I_JTRY = 370,
     I_PTRY = 371,
     I_RIS = 372,
     I_LDRF = 373,
     I_ONCE = 374,
     I_BAND = 375,
     I_BOR = 376,
     I_BXOR = 377,
     I_BNOT = 378,
     I_MODS = 379,
     I_AND = 380,
     I_OR = 381,
     I_ANDS = 382,
     I_ORS = 383,
     I_XORS = 384,
     I_NOTS = 385,
     I_HAS = 386,
     I_HASN = 387,
     I_GIVE = 388,
     I_GIVN = 389,
     I_IN = 390,
     I_NOIN = 391,
     I_PROV = 392,
     I_END = 393,
     I_PEEK = 394,
     I_PSIN = 395,
     I_PASS = 396,
     I_FORI = 397,
     I_FORN = 398,
     I_SHR = 399,
     I_SHL = 400,
     I_SHRS = 401,
     I_SHLS = 402,
     I_LDVR = 403,
     I_LDPR = 404,
     I_LSB = 405,
     I_INDI = 406,
     I_STEX = 407,
     I_TRAC = 408,
     I_WRT = 409
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
#define DSWITCH 283
#define DSELECT 284
#define DCASE 285
#define DENDSWITCH 286
#define DLINE 287
#define DSTRING 288
#define DCSTRING 289
#define DHAS 290
#define DHASNT 291
#define DINHERIT 292
#define DINSTANCE 293
#define SYMBOL 294
#define EXPORT 295
#define LABEL 296
#define INTEGER 297
#define REG_A 298
#define REG_B 299
#define REG_S1 300
#define REG_S2 301
#define NUMERIC 302
#define STRING 303
#define STRING_ID 304
#define TRUE_TOKEN 305
#define FALSE_TOKEN 306
#define I_LD 307
#define I_LNIL 308
#define NIL 309
#define I_ADD 310
#define I_SUB 311
#define I_MUL 312
#define I_DIV 313
#define I_MOD 314
#define I_POW 315
#define I_ADDS 316
#define I_SUBS 317
#define I_MULS 318
#define I_DIVS 319
#define I_POWS 320
#define I_INC 321
#define I_DEC 322
#define I_NEG 323
#define I_NOT 324
#define I_RET 325
#define I_RETV 326
#define I_RETA 327
#define I_FORK 328
#define I_PUSH 329
#define I_PSHR 330
#define I_PSHN 331
#define I_POP 332
#define I_LDV 333
#define I_LDVT 334
#define I_STV 335
#define I_STVR 336
#define I_STVS 337
#define I_LDP 338
#define I_LDPT 339
#define I_STP 340
#define I_STPR 341
#define I_STPS 342
#define I_TRAV 343
#define I_TRAN 344
#define I_TRAL 345
#define I_IPOP 346
#define I_XPOP 347
#define I_GENA 348
#define I_GEND 349
#define I_GENR 350
#define I_GEOR 351
#define I_JMP 352
#define I_IFT 353
#define I_IFF 354
#define I_BOOL 355
#define I_EQ 356
#define I_NEQ 357
#define I_GT 358
#define I_GE 359
#define I_LT 360
#define I_LE 361
#define I_UNPK 362
#define I_UNPS 363
#define I_CALL 364
#define I_INST 365
#define I_SWCH 366
#define I_SELE 367
#define I_NOP 368
#define I_TRY 369
#define I_JTRY 370
#define I_PTRY 371
#define I_RIS 372
#define I_LDRF 373
#define I_ONCE 374
#define I_BAND 375
#define I_BOR 376
#define I_BXOR 377
#define I_BNOT 378
#define I_MODS 379
#define I_AND 380
#define I_OR 381
#define I_ANDS 382
#define I_ORS 383
#define I_XORS 384
#define I_NOTS 385
#define I_HAS 386
#define I_HASN 387
#define I_GIVE 388
#define I_GIVN 389
#define I_IN 390
#define I_NOIN 391
#define I_PROV 392
#define I_END 393
#define I_PEEK 394
#define I_PSIN 395
#define I_PASS 396
#define I_FORI 397
#define I_FORN 398
#define I_SHR 399
#define I_SHL 400
#define I_SHRS 401
#define I_SHLS 402
#define I_LDVR 403
#define I_LDPR 404
#define I_LSB 405
#define I_INDI 406
#define I_STEX 407
#define I_TRAC 408
#define I_WRT 409




/* Copy the first part of user declarations.  */
#line 23 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"

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
#line 460 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"

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
#define YYLAST   1511

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  155
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  122
/* YYNRULES -- Number of rules.  */
#define YYNRULES  437
/* YYNRULES -- Number of states.  */
#define YYNSTATES  800

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   409

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
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    67,    69,    72,    76,    81,    86,    92,    96,   101,
     105,   110,   114,   118,   121,   125,   129,   134,   136,   139,
     144,   149,   154,   159,   164,   169,   174,   179,   184,   191,
     193,   197,   201,   205,   208,   211,   216,   222,   226,   231,
     234,   238,   241,   243,   244,   249,   252,   256,   259,   262,
     265,   267,   271,   273,   277,   280,   281,   283,   287,   289,
     291,   292,   294,   296,   298,   300,   302,   304,   306,   308,
     310,   312,   314,   316,   318,   320,   322,   324,   326,   328,
     330,   332,   334,   336,   338,   340,   342,   344,   346,   348,
     350,   352,   354,   356,   358,   360,   362,   364,   366,   368,
     370,   372,   374,   376,   378,   380,   382,   384,   386,   388,
     390,   392,   394,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   414,   416,   418,   420,   422,   424,   426,   428,
     430,   432,   434,   436,   438,   440,   442,   444,   446,   448,
     450,   452,   454,   456,   458,   460,   462,   464,   466,   468,
     470,   472,   474,   476,   478,   480,   482,   484,   486,   488,
     490,   492,   494,   496,   501,   504,   509,   512,   515,   518,
     523,   526,   531,   534,   539,   542,   547,   550,   555,   558,
     563,   566,   571,   574,   579,   582,   587,   590,   595,   598,
     603,   606,   611,   614,   619,   622,   627,   630,   635,   638,
     643,   646,   649,   652,   655,   658,   661,   664,   667,   670,
     673,   676,   679,   684,   689,   692,   697,   702,   705,   710,
     713,   718,   721,   724,   726,   729,   732,   735,   738,   741,
     744,   747,   750,   753,   758,   763,   766,   773,   780,   783,
     790,   797,   800,   807,   814,   817,   822,   827,   830,   835,
     840,   843,   850,   857,   860,   867,   874,   877,   884,   891,
     894,   899,   904,   907,   914,   921,   924,   931,   938,   941,
     944,   947,   950,   953,   956,   959,   962,   965,   968,   973,
     976,   979,   982,   985,   988,   991,   994,   997,  1000,  1003,
    1008,  1013,  1016,  1021,  1026,  1029,  1034,  1039,  1042,  1045,
    1048,  1051,  1053,  1056,  1058,  1061,  1064,  1067,  1069,  1072,
    1075,  1078,  1080,  1083,  1090,  1093,  1100,  1103,  1107,  1113,
    1118,  1123,  1126,  1131,  1136,  1141,  1146,  1149,  1154,  1159,
    1164,  1169,  1172,  1177,  1182,  1187,  1192,  1195,  1198,  1201,
    1204,  1209,  1212,  1217,  1220,  1225,  1228,  1233,  1236,  1241,
    1244,  1249,  1252,  1257,  1260,  1263,  1266,  1271,  1276,  1279,
    1284,  1289,  1292,  1297,  1302,  1305,  1310,  1315,  1318,  1323,
    1326,  1331,  1334,  1339,  1342,  1345,  1348,  1351,  1354,  1361,
    1368,  1371,  1376,  1381,  1384,  1389,  1392,  1397,  1400,  1405,
    1408,  1413,  1416,  1421,  1424,  1429,  1432,  1437,  1440,  1443,
    1446,  1449,  1452,  1455,  1458,  1461,  1464,  1467
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     156,     0,    -1,    -1,   156,   157,    -1,     3,    -1,   166,
       3,    -1,   170,   174,     3,    -1,   170,     3,    -1,   174,
       3,    -1,     1,     3,    -1,    54,    -1,   159,    -1,   160,
      -1,   163,    -1,    39,    -1,   161,    -1,    43,    -1,    44,
      -1,    45,    -1,    46,    -1,    54,    -1,   163,    -1,    47,
      -1,    50,    -1,    51,    -1,   164,    -1,    49,    -1,    48,
      -1,    42,    -1,    49,    -1,    48,    -1,     7,    -1,    26,
       4,    -1,     8,     4,   173,    -1,     8,     4,   173,    40,
      -1,     9,     4,   162,   173,    -1,     9,     4,   162,   173,
      40,    -1,    10,     4,   162,    -1,    10,     4,   162,    40,
      -1,    11,     4,   173,    -1,    11,     4,   173,    40,    -1,
      12,     4,   173,    -1,    13,     4,   173,    -1,    14,     4,
      -1,    14,     4,    40,    -1,    16,     4,   173,    -1,    16,
       4,   173,    40,    -1,    15,    -1,    27,     4,    -1,    28,
     160,     5,     4,    -1,    28,   160,     5,    42,    -1,    29,
     160,     5,     4,    -1,    29,   160,     5,    42,    -1,    30,
      54,     5,     4,    -1,    30,    42,     5,     4,    -1,    30,
      48,     5,     4,    -1,    30,    49,     5,     4,    -1,    30,
      39,     5,     4,    -1,    30,    42,     6,    42,     5,     4,
      -1,    31,    -1,    18,     4,   162,    -1,    18,     4,    39,
      -1,    19,     4,    39,    -1,    35,   168,    -1,    36,   169,
      -1,    38,    39,     4,   173,    -1,    38,    39,     4,   173,
      40,    -1,    20,     4,   173,    -1,    20,     4,   173,    40,
      -1,    21,     4,    -1,    21,     4,    40,    -1,    22,    39,
      -1,    23,    -1,    -1,    37,    39,   167,   171,    -1,    24,
       4,    -1,    25,     4,   173,    -1,    32,    42,    -1,    33,
      48,    -1,    34,    48,    -1,    39,    -1,   168,     5,    39,
      -1,    39,    -1,   169,     5,    39,    -1,    41,     6,    -1,
      -1,   172,    -1,   171,     5,   172,    -1,   162,    -1,    39,
      -1,    -1,    42,    -1,   175,    -1,   177,    -1,   178,    -1,
     180,    -1,   182,    -1,   184,    -1,   186,    -1,   187,    -1,
     179,    -1,   181,    -1,   183,    -1,   185,    -1,   195,    -1,
     196,    -1,   188,    -1,   189,    -1,   190,    -1,   191,    -1,
     192,    -1,   193,    -1,   197,    -1,   198,    -1,   203,    -1,
     204,    -1,   205,    -1,   207,    -1,   208,    -1,   209,    -1,
     210,    -1,   211,    -1,   212,    -1,   213,    -1,   214,    -1,
     215,    -1,   217,    -1,   216,    -1,   218,    -1,   219,    -1,
     220,    -1,   221,    -1,   222,    -1,   223,    -1,   224,    -1,
     225,    -1,   227,    -1,   228,    -1,   229,    -1,   230,    -1,
     231,    -1,   201,    -1,   202,    -1,   199,    -1,   200,    -1,
     233,    -1,   235,    -1,   234,    -1,   236,    -1,   194,    -1,
     237,    -1,   232,    -1,   226,    -1,   239,    -1,   240,    -1,
     176,    -1,   238,    -1,   243,    -1,   244,    -1,   245,    -1,
     246,    -1,   247,    -1,   248,    -1,   249,    -1,   250,    -1,
     251,    -1,   254,    -1,   252,    -1,   253,    -1,   255,    -1,
     256,    -1,   257,    -1,   258,    -1,   259,    -1,   260,    -1,
     261,    -1,   242,    -1,   206,    -1,   262,    -1,   263,    -1,
     264,    -1,   265,    -1,   266,    -1,   267,    -1,   268,    -1,
     269,    -1,   270,    -1,   271,    -1,   272,    -1,   273,    -1,
     274,    -1,   275,    -1,   276,    -1,    52,   160,     5,   158,
      -1,    52,     1,    -1,   118,   160,     5,   158,    -1,   118,
       1,    -1,    53,   160,    -1,    53,     1,    -1,    55,   158,
       5,   158,    -1,    55,     1,    -1,    61,   160,     5,   158,
      -1,    61,     1,    -1,    56,   158,     5,   158,    -1,    56,
       1,    -1,    62,   160,     5,   158,    -1,    62,     1,    -1,
      57,   158,     5,   158,    -1,    57,     1,    -1,    63,   160,
       5,   158,    -1,    63,     1,    -1,    58,   158,     5,   158,
      -1,    58,     1,    -1,    64,   160,     5,   158,    -1,    64,
       1,    -1,    59,   158,     5,   158,    -1,    59,     1,    -1,
      60,   158,     5,   158,    -1,    60,     1,    -1,   101,   158,
       5,   158,    -1,   101,     1,    -1,   102,   158,     5,   158,
      -1,   102,     1,    -1,   104,   158,     5,   158,    -1,   104,
       1,    -1,   103,   158,     5,   158,    -1,   103,     1,    -1,
     106,   158,     5,   158,    -1,   106,     1,    -1,   105,   158,
       5,   158,    -1,   105,     1,    -1,   114,     4,    -1,   114,
      42,    -1,   114,     1,    -1,    66,   160,    -1,    66,     1,
      -1,    67,   160,    -1,    67,     1,    -1,    68,   158,    -1,
      68,     1,    -1,    69,   158,    -1,    69,     1,    -1,   109,
      42,     5,   160,    -1,   109,    42,     5,     4,    -1,   109,
       1,    -1,   110,    42,     5,   160,    -1,   110,    42,     5,
       4,    -1,   110,     1,    -1,   107,   160,     5,   160,    -1,
     107,     1,    -1,   108,    42,     5,   160,    -1,   108,     1,
      -1,    74,   158,    -1,    76,    -1,    74,     1,    -1,    75,
     158,    -1,    75,     1,    -1,    77,   160,    -1,    77,     1,
      -1,   139,   160,    -1,   139,     1,    -1,    92,   160,    -1,
      92,     1,    -1,    78,   160,     5,   160,    -1,    78,   160,
       5,   164,    -1,    78,     1,    -1,    79,   160,     5,   160,
       5,   160,    -1,    79,   160,     5,   164,     5,   160,    -1,
      79,     1,    -1,    80,   160,     5,   160,     5,   158,    -1,
      80,   160,     5,   164,     5,   158,    -1,    80,     1,    -1,
      81,   160,     5,   160,     5,   158,    -1,    81,   160,     5,
     164,     5,   158,    -1,    81,     1,    -1,    82,   160,     5,
     160,    -1,    82,   160,     5,   164,    -1,    82,     1,    -1,
      83,   160,     5,   160,    -1,    83,   160,     5,   164,    -1,
      83,     1,    -1,    84,   160,     5,   160,     5,   160,    -1,
      84,   160,     5,   164,     5,   160,    -1,    84,     1,    -1,
      85,   160,     5,   160,     5,   158,    -1,    85,   160,     5,
     164,     5,   158,    -1,    85,     1,    -1,    86,   160,     5,
     160,     5,   160,    -1,    86,   160,     5,   164,     5,   160,
      -1,    86,     1,    -1,    87,   160,     5,   160,    -1,    87,
     160,     5,   164,    -1,    87,     1,    -1,    88,    42,     5,
     158,     5,   158,    -1,    88,     4,     5,   158,     5,   158,
      -1,    88,     1,    -1,    89,     4,     5,     4,     5,    42,
      -1,    89,    42,     5,    42,     5,    42,    -1,    89,     1,
      -1,    90,    42,    -1,    90,     4,    -1,    90,     1,    -1,
      91,    42,    -1,    91,     1,    -1,    93,    42,    -1,    93,
       1,    -1,    94,    42,    -1,    94,     1,    -1,    95,   159,
       5,   159,    -1,    95,     1,    -1,    96,   159,    -1,    96,
       1,    -1,   117,   158,    -1,   117,     1,    -1,    97,     4,
      -1,    97,    42,    -1,    97,     1,    -1,   100,   158,    -1,
     100,     1,    -1,    98,     4,     5,   158,    -1,    98,    42,
       5,   158,    -1,    98,     1,    -1,    99,     4,     5,   158,
      -1,    99,    42,     5,   158,    -1,    99,     1,    -1,    73,
      42,     5,     4,    -1,    73,    42,     5,    42,    -1,    73,
       1,    -1,   115,     4,    -1,   115,    42,    -1,   115,     1,
      -1,    70,    -1,    70,     1,    -1,    72,    -1,    72,     1,
      -1,    71,   158,    -1,    71,     1,    -1,   113,    -1,   113,
       1,    -1,   116,    42,    -1,   116,     1,    -1,   138,    -1,
     138,     1,    -1,   111,    42,     5,   160,     5,   241,    -1,
     111,     1,    -1,   112,    42,     5,   160,     5,   241,    -1,
     112,     1,    -1,    42,     5,     4,    -1,   241,     5,    42,
       5,     4,    -1,   119,    42,     5,   158,    -1,   119,     4,
       5,   158,    -1,   119,     1,    -1,   120,    39,     5,    39,
      -1,   120,    39,     5,    42,    -1,   120,    42,     5,    39,
      -1,   120,    42,     5,    42,    -1,   120,     1,    -1,   121,
      39,     5,    39,    -1,   121,    39,     5,    42,    -1,   121,
      42,     5,    39,    -1,   121,    42,     5,    42,    -1,   121,
       1,    -1,   122,    39,     5,    39,    -1,   122,    39,     5,
      42,    -1,   122,    42,     5,    39,    -1,   122,    42,     5,
      42,    -1,   122,     1,    -1,   123,    39,    -1,   123,    42,
      -1,   123,     1,    -1,   125,   158,     5,   158,    -1,   125,
       1,    -1,   126,   158,     5,   158,    -1,   126,     1,    -1,
     127,   160,     5,   159,    -1,   127,     1,    -1,   128,   160,
       5,   159,    -1,   128,     1,    -1,   129,   160,     5,   159,
      -1,   129,     1,    -1,   124,   160,     5,   159,    -1,   124,
       1,    -1,    65,   160,     5,   159,    -1,    65,     1,    -1,
     130,   160,    -1,   130,     1,    -1,   131,   160,     5,   160,
      -1,   131,   160,     5,    42,    -1,   131,     1,    -1,   132,
     160,     5,   160,    -1,   132,   160,     5,    42,    -1,   132,
       1,    -1,   133,   160,     5,   160,    -1,   133,   160,     5,
      42,    -1,   133,     1,    -1,   134,   160,     5,   160,    -1,
     134,   160,     5,    42,    -1,   134,     1,    -1,   135,   158,
       5,   159,    -1,   135,     1,    -1,   136,   158,     5,   159,
      -1,   136,     1,    -1,   137,   158,     5,   159,    -1,   137,
       1,    -1,   140,   160,    -1,   140,     1,    -1,   141,   160,
      -1,   141,     1,    -1,   142,     4,     5,    39,     5,   158,
      -1,   142,    42,     5,    39,     5,   158,    -1,   142,     1,
      -1,   143,     4,     5,    39,    -1,   143,    42,     5,    39,
      -1,   143,     1,    -1,   144,   158,     5,   158,    -1,   144,
       1,    -1,   145,   158,     5,   158,    -1,   145,     1,    -1,
     146,   158,     5,   158,    -1,   146,     1,    -1,   147,   158,
       5,   158,    -1,   147,     1,    -1,   148,   158,     5,   158,
      -1,   148,     1,    -1,   149,   158,     5,   158,    -1,   149,
       1,    -1,   150,   158,     5,   158,    -1,   150,     1,    -1,
     151,   165,    -1,   151,   160,    -1,   151,     1,    -1,   152,
     165,    -1,   152,   160,    -1,   152,     1,    -1,   153,   158,
      -1,   153,     1,    -1,   154,   158,    -1,   154,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   231,   231,   233,   237,   238,   239,   240,   241,   242,
     245,   245,   246,   246,   247,   247,   248,   248,   248,   248,
     250,   250,   251,   251,   251,   251,   252,   252,   252,   253,
     253,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   298,   299,   300,   301,   302,   307,
     316,   317,   321,   322,   325,   327,   329,   330,   334,   335,
     339,   340,   344,   345,   346,   347,   348,   349,   350,   351,
     352,   353,   354,   355,   356,   357,   358,   359,   360,   361,
     362,   363,   364,   365,   366,   367,   368,   369,   370,   371,
     372,   373,   374,   375,   376,   377,   378,   379,   380,   381,
     382,   383,   384,   385,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   406,   407,   408,   409,   410,   411,
     412,   413,   414,   415,   416,   417,   418,   419,   420,   421,
     422,   423,   424,   425,   426,   427,   428,   429,   430,   431,
     432,   433,   434,   435,   436,   437,   438,   439,   440,   441,
     442,   443,   444,   448,   449,   453,   454,   458,   459,   463,
     464,   468,   469,   474,   475,   479,   480,   484,   485,   489,
     490,   495,   496,   500,   501,   505,   506,   510,   511,   516,
     517,   521,   522,   526,   527,   531,   532,   536,   537,   541,
     542,   546,   547,   548,   552,   553,   557,   558,   562,   563,
     567,   568,   572,   573,   574,   578,   579,   580,   584,   585,
     589,   590,   595,   596,   597,   601,   602,   607,   608,   612,
     613,   617,   618,   623,   624,   625,   629,   630,   631,   635,
     636,   637,   641,   642,   643,   647,   648,   649,   653,   654,
     655,   659,   660,   661,   665,   666,   667,   671,   672,   673,
     677,   678,   679,   683,   684,   685,   689,   690,   691,   695,
     696,   697,   701,   702,   706,   707,   711,   712,   716,   717,
     721,   722,   726,   727,   731,   732,   733,   737,   738,   742,
     743,   744,   748,   749,   750,   755,   756,   757,   761,   762,
     763,   767,   768,   772,   773,   777,   778,   782,   783,   787,
     788,   792,   793,   797,   798,   802,   803,   807,   816,   825,
     826,   827,   831,   832,   833,   834,   835,   839,   840,   841,
     842,   843,   847,   848,   849,   850,   851,   855,   856,   857,
     861,   862,   866,   867,   871,   872,   876,   877,   881,   882,
     886,   887,   891,   892,   896,   897,   901,   902,   903,   907,
     908,   909,   913,   914,   915,   919,   920,   921,   926,   927,
     931,   932,   936,   937,   941,   942,   946,   947,   951,   952,
     953,   957,   958,   959,   963,   964,   968,   969,   973,   974,
     978,   979,   983,   984,   988,   989,   993,   994,   998,   999,
    1000,  1004,  1005,  1006,  1010,  1011,  1015,  1016
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
  "DMODULE", "DLOAD", "DSWITCH", "DSELECT", "DCASE", "DENDSWITCH", "DLINE",
  "DSTRING", "DCSTRING", "DHAS", "DHASNT", "DINHERIT", "DINSTANCE",
  "SYMBOL", "EXPORT", "LABEL", "INTEGER", "REG_A", "REG_B", "REG_S1",
  "REG_S2", "NUMERIC", "STRING", "STRING_ID", "TRUE_TOKEN", "FALSE_TOKEN",
  "I_LD", "I_LNIL", "NIL", "I_ADD", "I_SUB", "I_MUL", "I_DIV", "I_MOD",
  "I_POW", "I_ADDS", "I_SUBS", "I_MULS", "I_DIVS", "I_POWS", "I_INC",
  "I_DEC", "I_NEG", "I_NOT", "I_RET", "I_RETV", "I_RETA", "I_FORK",
  "I_PUSH", "I_PSHR", "I_PSHN", "I_POP", "I_LDV", "I_LDVT", "I_STV",
  "I_STVR", "I_STVS", "I_LDP", "I_LDPT", "I_STP", "I_STPR", "I_STPS",
  "I_TRAV", "I_TRAN", "I_TRAL", "I_IPOP", "I_XPOP", "I_GENA", "I_GEND",
  "I_GENR", "I_GEOR", "I_JMP", "I_IFT", "I_IFF", "I_BOOL", "I_EQ", "I_NEQ",
  "I_GT", "I_GE", "I_LT", "I_LE", "I_UNPK", "I_UNPS", "I_CALL", "I_INST",
  "I_SWCH", "I_SELE", "I_NOP", "I_TRY", "I_JTRY", "I_PTRY", "I_RIS",
  "I_LDRF", "I_ONCE", "I_BAND", "I_BOR", "I_BXOR", "I_BNOT", "I_MODS",
  "I_AND", "I_OR", "I_ANDS", "I_ORS", "I_XORS", "I_NOTS", "I_HAS",
  "I_HASN", "I_GIVE", "I_GIVN", "I_IN", "I_NOIN", "I_PROV", "I_END",
  "I_PEEK", "I_PSIN", "I_PASS", "I_FORI", "I_FORN", "I_SHR", "I_SHL",
  "I_SHRS", "I_SHLS", "I_LDVR", "I_LDPR", "I_LSB", "I_INDI", "I_STEX",
  "I_TRAC", "I_WRT", "$accept", "input", "line", "xoperand", "operand",
  "op_variable", "op_register", "x_op_immediate", "op_immediate",
  "op_scalar", "op_string", "directive", "@1", "has_symlist",
  "hasnt_symlist", "label", "inherit_param_list", "inherit_param",
  "def_line", "instruction", "inst_ld", "inst_ldrf", "inst_ldnil",
  "inst_add", "inst_adds", "inst_sub", "inst_subs", "inst_mul",
  "inst_muls", "inst_div", "inst_divs", "inst_mod", "inst_pow", "inst_eq",
  "inst_ne", "inst_ge", "inst_gt", "inst_le", "inst_lt", "inst_try",
  "inst_inc", "inst_dec", "inst_neg", "inst_not", "inst_call", "inst_inst",
  "inst_unpk", "inst_unps", "inst_push", "inst_pshr", "inst_pop",
  "inst_peek", "inst_xpop", "inst_ldv", "inst_ldvt", "inst_stv",
  "inst_stvr", "inst_stvs", "inst_ldp", "inst_ldpt", "inst_stp",
  "inst_stpr", "inst_stps", "inst_trav", "inst_tran", "inst_tral",
  "inst_ipop", "inst_gena", "inst_gend", "inst_genr", "inst_geor",
  "inst_ris", "inst_jmp", "inst_bool", "inst_ift", "inst_iff", "inst_fork",
  "inst_jtry", "inst_ret", "inst_reta", "inst_retval", "inst_nop",
  "inst_ptry", "inst_end", "inst_swch", "inst_sele", "switch_list",
  "inst_once", "inst_band", "inst_bor", "inst_bxor", "inst_bnot",
  "inst_and", "inst_or", "inst_ands", "inst_ors", "inst_xors", "inst_mods",
  "inst_pows", "inst_nots", "inst_has", "inst_hasn", "inst_give",
  "inst_givn", "inst_in", "inst_noin", "inst_prov", "inst_psin",
  "inst_pass", "inst_fori", "inst_forn", "inst_shr", "inst_shl",
  "inst_shrs", "inst_shls", "inst_ldvr", "inst_ldpr", "inst_lsb",
  "inst_indi", "inst_stex", "inst_trac", "inst_wrt", 0
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
     405,   406,   407,   408,   409
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   155,   156,   156,   157,   157,   157,   157,   157,   157,
     158,   158,   159,   159,   160,   160,   161,   161,   161,   161,
     162,   162,   163,   163,   163,   163,   164,   164,   164,   165,
     165,   166,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   167,   166,   166,   166,   166,   166,   166,
     168,   168,   169,   169,   170,   171,   171,   171,   172,   172,
     173,   173,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   175,   175,   176,   176,   177,   177,   178,
     178,   179,   179,   180,   180,   181,   181,   182,   182,   183,
     183,   184,   184,   185,   185,   186,   186,   187,   187,   188,
     188,   189,   189,   190,   190,   191,   191,   192,   192,   193,
     193,   194,   194,   194,   195,   195,   196,   196,   197,   197,
     198,   198,   199,   199,   199,   200,   200,   200,   201,   201,
     202,   202,   203,   203,   203,   204,   204,   205,   205,   206,
     206,   207,   207,   208,   208,   208,   209,   209,   209,   210,
     210,   210,   211,   211,   211,   212,   212,   212,   213,   213,
     213,   214,   214,   214,   215,   215,   215,   216,   216,   216,
     217,   217,   217,   218,   218,   218,   219,   219,   219,   220,
     220,   220,   221,   221,   222,   222,   223,   223,   224,   224,
     225,   225,   226,   226,   227,   227,   227,   228,   228,   229,
     229,   229,   230,   230,   230,   231,   231,   231,   232,   232,
     232,   233,   233,   234,   234,   235,   235,   236,   236,   237,
     237,   238,   238,   239,   239,   240,   240,   241,   241,   242,
     242,   242,   243,   243,   243,   243,   243,   244,   244,   244,
     244,   244,   245,   245,   245,   245,   245,   246,   246,   246,
     247,   247,   248,   248,   249,   249,   250,   250,   251,   251,
     252,   252,   253,   253,   254,   254,   255,   255,   255,   256,
     256,   256,   257,   257,   257,   258,   258,   258,   259,   259,
     260,   260,   261,   261,   262,   262,   263,   263,   264,   264,
     264,   265,   265,   265,   266,   266,   267,   267,   268,   268,
     269,   269,   270,   270,   271,   271,   272,   272,   273,   273,
     273,   274,   274,   274,   275,   275,   276,   276
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     3,     4,     4,     5,     3,     4,     3,
       4,     3,     3,     2,     3,     3,     4,     1,     2,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     6,     1,
       3,     3,     3,     2,     2,     4,     5,     3,     4,     2,
       3,     2,     1,     0,     4,     2,     3,     2,     2,     2,
       1,     3,     1,     3,     2,     0,     1,     3,     1,     1,
       0,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     2,     4,     2,     2,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     4,     4,     2,     4,     4,     2,     4,     2,
       4,     2,     2,     1,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     4,     4,     2,     6,     6,     2,     6,
       6,     2,     6,     6,     2,     4,     4,     2,     4,     4,
       2,     6,     6,     2,     6,     6,     2,     6,     6,     2,
       4,     4,     2,     6,     6,     2,     6,     6,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     4,
       4,     2,     4,     4,     2,     4,     4,     2,     2,     2,
       2,     1,     2,     1,     2,     2,     2,     1,     2,     2,
       2,     1,     2,     6,     2,     6,     2,     3,     5,     4,
       4,     2,     4,     4,     4,     4,     2,     4,     4,     4,
       4,     2,     4,     4,     4,     4,     2,     2,     2,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     2,     2,     4,     4,     2,     4,
       4,     2,     4,     4,     2,     4,     4,     2,     4,     2,
       4,     2,     4,     2,     2,     2,     2,     2,     6,     6,
       2,     4,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    31,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,     0,     0,     0,
      72,     0,     0,     0,     0,     0,     0,     0,    59,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     253,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     3,
       0,     0,     0,    92,   155,    93,    94,   100,    95,   101,
      96,   102,    97,   103,    98,    99,   106,   107,   108,   109,
     110,   111,   149,   104,   105,   112,   113,   143,   144,   141,
     142,   114,   115,   116,   177,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   127,   126,   128,   129,   130,   131,
     132,   133,   134,   135,   152,   136,   137,   138,   139,   140,
     151,   145,   147,   146,   148,   150,   156,   153,   154,   176,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   167,
     168,   166,   169,   170,   171,   172,   173,   174,   175,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,     9,    90,     0,     0,    90,    90,
      90,    43,    90,     0,     0,    90,    69,    71,    75,    90,
      32,    48,    14,    16,    17,    18,    19,     0,    15,     0,
       0,     0,     0,     0,     0,    77,    78,    79,    80,    63,
      82,    64,    73,     0,    84,   194,     0,   198,   197,   200,
      28,    22,    27,    26,    23,    24,    10,     0,    11,    12,
      13,    25,   204,     0,   208,     0,   212,     0,   216,     0,
     218,     0,   202,     0,   206,     0,   210,     0,   214,     0,
     383,     0,   235,   234,   237,   236,   239,   238,   241,   240,
     332,   336,   335,   334,   327,     0,   254,   252,   256,   255,
     258,   257,   265,     0,   268,     0,   271,     0,   274,     0,
     277,     0,   280,     0,   283,     0,   286,     0,   289,     0,
     292,     0,   295,     0,     0,   298,     0,     0,   301,   300,
     299,   303,   302,   262,   261,   305,   304,   307,   306,   309,
       0,   311,   310,   316,   314,   315,   321,     0,     0,   324,
       0,     0,   318,   317,   220,     0,   222,     0,   226,     0,
     224,     0,   230,     0,   228,     0,   249,     0,   251,     0,
     244,     0,   247,     0,   344,     0,   346,     0,   338,   233,
     231,   232,   330,   328,   329,   340,   339,   313,   312,   196,
       0,   351,     0,     0,   356,     0,     0,   361,     0,     0,
     366,     0,     0,   369,   367,   368,   381,     0,   371,     0,
     373,     0,   375,     0,   377,     0,   379,     0,   385,   384,
     388,     0,   391,     0,   394,     0,   397,     0,   399,     0,
     401,     0,   403,     0,   342,   260,   259,   405,   404,   407,
     406,   410,     0,     0,   413,     0,     0,   415,     0,   417,
       0,   419,     0,   421,     0,   423,     0,   425,     0,   427,
       0,   430,    30,    29,   429,   428,   433,   432,   431,   435,
     434,   437,   436,     5,     7,     0,     8,    91,    33,    20,
      90,    21,    37,    39,    41,    42,    44,    45,    61,    60,
      62,    67,    70,    76,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    85,    90,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,    34,    35,    38,    40,    46,    68,
      49,    50,    51,    52,    57,    54,     0,    55,    56,    53,
      81,    83,    89,    88,    74,    86,    65,   193,   199,   203,
     207,   211,   215,   217,   201,   205,   209,   213,   382,   325,
     326,   263,   264,     0,     0,     0,     0,     0,     0,   275,
     276,   278,   279,     0,     0,     0,     0,     0,     0,   290,
     291,     0,     0,     0,     0,   308,   319,   320,   322,   323,
     219,   221,   225,   223,   229,   227,   248,   250,   243,   242,
     246,   245,     0,     0,   195,   350,   349,   352,   353,   354,
     355,   357,   358,   359,   360,   362,   363,   364,   365,   380,
     370,   372,   374,   376,   378,   387,   386,   390,   389,   393,
     392,   396,   395,   398,   400,   402,     0,     0,   411,   412,
     414,   416,   418,   420,   422,   424,   426,    36,     0,     0,
      66,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    58,    87,   266,   267,   269,   270,   272,   273,   281,
     282,   284,   285,   287,   288,   294,   293,   296,   297,     0,
     343,   345,   408,   409,     0,     0,   347,     0,     0,   348
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   139,   297,   298,   299,   268,   643,   300,   301,
     505,   140,   544,   279,   281,   141,   644,   645,   518,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   790,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -245
static const yytype_int16 yypact[] =
{
    -245,   303,  -245,    25,  -245,  -245,     4,    10,    60,   113,
     148,   152,   157,  -245,   164,   173,   240,   243,   259,    -5,
    -245,   272,   287,   298,   305,   611,   611,   579,  -245,    55,
     140,   306,   307,   312,   313,   314,   351,    17,   249,    79,
     158,   171,   184,   210,   223,   304,   554,   672,   685,   761,
    1168,  1181,   236,   464,    32,   477,   106,     5,   621,   645,
    -245,  1194,  1207,  1215,  1227,  1235,  1243,  1254,  1262,  1274,
    1282,  1290,    15,    26,    28,    16,  1301,    53,    64,    93,
     528,   111,   154,   247,   658,   709,   722,   742,   934,   947,
     960,  1309,    66,   300,   503,   545,   603,   295,   459,   460,
     605,   973,  1321,   475,   491,   546,   547,   738,  1329,   986,
     999,  1337,  1348,  1356,  1368,  1376,  1384,  1395,  1403,  1012,
    1025,  1038,   296,  1415,  1423,  1431,   490,   550,  1051,  1064,
    1077,  1090,  1103,  1116,  1129,    30,   130,  1142,  1155,  -245,
     463,   780,   470,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   435,   895,   895,   435,   435,
     435,   445,   435,   420,   447,   435,   448,  -245,  -245,   435,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   484,  -245,   485,
     539,   302,   551,   552,   553,  -245,  -245,  -245,  -245,   561,
    -245,   563,  -245,   492,  -245,  -245,   564,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   600,  -245,  -245,
    -245,  -245,  -245,   615,  -245,   624,  -245,   627,  -245,   629,
    -245,   643,  -245,   644,  -245,   653,  -245,   669,  -245,   671,
    -245,   680,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   693,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,   708,  -245,   714,  -245,   715,  -245,   716,
    -245,   717,  -245,   720,  -245,   729,  -245,   730,  -245,   733,
    -245,   735,  -245,   736,   737,  -245,   739,   740,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
     741,  -245,  -245,  -245,  -245,  -245,  -245,   744,   745,  -245,
     769,   770,  -245,  -245,  -245,   773,  -245,   774,  -245,   777,
    -245,   789,  -245,   790,  -245,   792,  -245,   793,  -245,   794,
    -245,   796,  -245,   797,  -245,   798,  -245,   803,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
     804,  -245,   805,   806,  -245,   807,   808,  -245,   813,   814,
    -245,   817,   818,  -245,  -245,  -245,  -245,   821,  -245,   822,
    -245,   823,  -245,   824,  -245,   829,  -245,   931,  -245,  -245,
    -245,   933,  -245,   935,  -245,   936,  -245,   970,  -245,  1205,
    -245,  1210,  -245,  1212,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  1213,  1214,  -245,  1216,  1217,  -245,  1218,  -245,
    1224,  -245,  1225,  -245,  1226,  -245,  1229,  -245,  1230,  -245,
    1236,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   711,  -245,  -245,   455,  -245,
     435,  -245,   510,   513,  -245,  -245,  -245,   707,  -245,  -245,
    -245,   899,  -245,  -245,    11,    92,  1228,  1238,  1201,  1241,
    1244,  1245,  1208,  1223,   433,   435,   908,   908,   908,   908,
     908,   908,   908,   908,   908,   908,   908,   921,   548,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   197,   908,
     908,  1252,  1222,   921,   908,   908,   908,   908,   908,   908,
     908,   908,   908,   908,   611,   611,   454,   580,   611,   611,
     908,   908,   908,   -32,    42,    43,   115,   144,   145,   921,
     908,   908,   921,   921,   921,   121,  1449,  1457,  1465,   921,
     921,   921,  1237,  1246,  1251,  1253,   908,   908,   908,   908,
     908,   908,   908,  -245,  -245,  1255,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  1260,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1263,  -245,  1256,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  1264,  1272,  1279,  1289,  1298,  1299,  -245,
    -245,  -245,  -245,  1304,  1306,  1307,  1310,  1311,  1318,  -245,
    -245,  1319,  1326,  1327,  1332,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  1334,  1336,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  1338,  1345,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  1347,   433,
    -245,   611,   611,   908,   908,   908,   908,   611,   611,   908,
     908,   611,   611,   908,   908,  1300,  1314,  1316,  1316,   908,
     908,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  1354,
    1357,  1357,  -245,  -245,  1359,  1328,  -245,  1366,  1374,  -245
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -245,  -245,  -245,    61,    -8,   -25,  -245,  -242,  -244,   919,
    1121,  -245,  -245,  -245,  -245,  -245,  -245,   518,  -200,  1173,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,   593,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -342
static const yytype_int16 yytable[] =
{
     267,   269,   521,   521,   520,   522,   334,   707,   245,   521,
     708,   529,   286,   288,   246,   630,   362,   371,   285,   363,
     313,   315,   317,   319,   321,   323,   325,   365,   244,   368,
     366,   501,   369,   330,   257,  -331,   341,   343,   345,   347,
     349,   351,   353,   355,   357,   359,   361,   335,   523,   524,
     525,   374,   527,   631,   375,   531,   262,   364,   372,   533,
     263,   264,   265,   266,   247,   377,   407,   408,   367,   262,
     370,   380,   382,   263,   264,   265,   266,   430,   502,   503,
     289,   709,   711,   447,   710,   712,   453,   455,   457,   459,
     461,   463,   465,   467,   379,   376,   632,   275,   476,   478,
     480,   303,   305,   307,   309,   311,   378,   333,   409,  -333,
     504,   507,   383,   327,   329,   384,   332,   248,   262,   337,
     339,   290,   263,   264,   265,   266,   291,   292,   293,   294,
     295,   506,   262,   296,   633,   290,   263,   264,   265,   266,
     291,   292,   293,   294,   295,   393,   395,   397,   399,   401,
     403,   405,   249,   385,   713,   386,   250,   714,   387,   302,
     262,   251,   428,   725,   263,   264,   265,   266,   252,   262,
     449,   451,   304,   263,   264,   265,   266,   253,   502,   503,
     469,   471,   473,   715,   717,   306,   716,   718,   276,   488,
     490,   492,   494,   496,   498,   500,   388,   262,   510,   512,
     290,   263,   264,   265,   266,   291,   292,   293,   294,   295,
     262,   308,   296,   290,   263,   264,   265,   266,   291,   292,
     293,   294,   295,   262,   310,   296,   290,   263,   264,   265,
     266,   291,   292,   293,   294,   295,   262,   326,   296,   290,
     263,   264,   265,   266,   254,   292,   293,   255,   389,   262,
     287,   390,   290,   263,   264,   265,   266,   291,   292,   293,
     294,   295,   262,   256,   296,   290,   263,   264,   265,   266,
     291,   292,   293,   294,   295,   262,   258,   296,   290,   263,
     264,   265,   266,   291,   292,   293,   294,   295,   262,   391,
     296,   259,   263,   264,   265,   266,   418,   474,  -337,  -341,
     521,   410,   260,     2,     3,   312,     4,   537,   538,   261,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
     625,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,   411,   262,    36,   646,   278,   263,   264,   265,
     266,   280,   282,   283,   277,    37,    38,   284,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   698,   528,
     419,   422,   290,   420,   423,   328,   513,   291,   292,   293,
     294,   295,   642,   516,   519,   290,   431,   517,   331,   432,
     291,   292,   293,   294,   295,   526,   530,   519,   532,   534,
     535,   481,   434,   262,   482,   624,   545,   263,   264,   265,
     266,   421,   424,   262,   412,   521,   290,   263,   264,   265,
     266,   291,   292,   293,   294,   295,   262,   433,   296,   290,
     263,   264,   265,   266,   291,   292,   293,   294,   295,   381,
     435,   296,   483,   436,   661,   663,   665,   667,   669,   671,
     673,   675,   677,   679,   536,   413,   414,   437,   440,   658,
     626,   484,   659,   627,   485,   314,   539,   540,   541,   696,
     697,   699,   701,   702,   703,   685,   542,   262,   543,   546,
     290,   263,   264,   265,   266,   291,   292,   293,   294,   295,
     726,   728,   730,   732,   700,   438,   441,   415,   439,   442,
     660,   719,   486,   262,   722,   723,   724,   263,   264,   265,
     266,   733,   734,   735,   416,   547,   425,   647,   648,   649,
     650,   651,   652,   653,   654,   655,   656,   657,   270,   262,
     548,   271,   336,   263,   264,   265,   266,   272,   273,   549,
     681,   682,   550,   274,   551,   686,   687,   688,   689,   690,
     691,   692,   693,   694,   695,   417,   338,   426,   552,   553,
     262,   704,   705,   706,   263,   264,   265,   266,   554,   392,
     262,   720,   721,   290,   263,   264,   265,   266,   291,   292,
     293,   294,   295,   316,   555,   296,   556,   740,   741,   742,
     743,   744,   745,   746,   262,   557,   318,   290,   263,   264,
     265,   266,   291,   292,   293,   294,   295,   262,   558,   296,
     290,   263,   264,   265,   266,   291,   292,   293,   294,   295,
     394,   262,   296,   559,   623,   263,   264,   265,   266,   560,
     561,   562,   563,   396,   262,   564,   773,   774,   263,   264,
     265,   266,   779,   780,   565,   566,   783,   784,   567,   443,
     568,   569,   570,   398,   571,   572,   573,   628,   262,   574,
     575,   290,   263,   264,   265,   266,   291,   292,   293,   294,
     295,   262,   320,   296,   290,   263,   264,   265,   266,   291,
     292,   293,   294,   295,   576,   577,   296,   444,   578,   579,
     445,   262,   580,   514,   290,   263,   264,   265,   266,   291,
     292,   293,   294,   295,   581,   582,   296,   583,   584,   585,
     262,   586,   587,   588,   263,   264,   265,   266,   589,   590,
     591,   592,   593,   594,   775,   776,   777,   778,   595,   596,
     781,   782,   597,   598,   785,   786,   599,   600,   601,   602,
     792,   793,    37,    38,   603,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   400,   604,   290,   605,   629,
     606,   607,   291,   292,   293,   294,   295,   262,   402,   519,
     290,   263,   264,   265,   266,   291,   292,   293,   294,   295,
     262,   404,   296,   290,   263,   264,   265,   266,   291,   292,
     293,   294,   295,   262,   427,   608,   290,   263,   264,   265,
     266,   291,   292,   293,   294,   295,   262,   448,   296,   290,
     263,   264,   265,   266,   291,   292,   293,   294,   295,   262,
     450,   296,   290,   263,   264,   265,   266,   291,   292,   293,
     294,   295,   262,   468,   296,   290,   263,   264,   265,   266,
     291,   292,   293,   294,   295,   262,   470,   296,   290,   263,
     264,   265,   266,   291,   292,   293,   294,   295,   262,   472,
     296,   290,   263,   264,   265,   266,   291,   292,   293,   294,
     295,   262,   487,   296,   290,   263,   264,   265,   266,   291,
     292,   293,   294,   295,   262,   489,   296,   290,   263,   264,
     265,   266,   291,   292,   293,   294,   295,   262,   491,   296,
     290,   263,   264,   265,   266,   291,   292,   293,   294,   295,
     262,   493,   296,   290,   263,   264,   265,   266,   291,   292,
     293,   294,   295,   262,   495,   296,   290,   263,   264,   265,
     266,   291,   292,   293,   294,   295,   262,   497,   296,   290,
     263,   264,   265,   266,   291,   292,   293,   294,   295,   262,
     499,   296,   290,   263,   264,   265,   266,   291,   292,   293,
     294,   295,   262,   509,   296,   290,   263,   264,   265,   266,
     291,   292,   293,   294,   295,   262,   511,   296,   290,   263,
     264,   265,   266,   291,   292,   293,   294,   295,   262,   322,
     296,   290,   263,   264,   265,   266,   291,   292,   293,   294,
     295,   262,   324,   296,   290,   263,   264,   265,   266,   291,
     292,   293,   294,   295,   262,   340,   296,   290,   263,   264,
     265,   266,   291,   292,   293,   294,   295,   262,   342,   296,
     609,   263,   264,   265,   266,   610,   344,   611,   612,   613,
     262,   614,   615,   616,   263,   264,   265,   266,   346,   617,
     618,   619,   634,   262,   620,   621,   348,   263,   264,   265,
     266,   622,   635,   636,   350,   637,   262,   640,   638,   639,
     263,   264,   265,   266,   262,   352,   683,   508,   263,   264,
     265,   266,   641,   354,   684,   748,   262,   772,   749,   751,
     263,   264,   265,   266,   262,   356,   736,   752,   263,   264,
     265,   266,   262,   358,   753,   737,   263,   264,   265,   266,
     738,   360,   739,   262,   754,   747,   750,   263,   264,   265,
     266,   262,   373,   755,   756,   263,   264,   265,   266,   757,
     406,   758,   759,   262,   515,   760,   761,   263,   264,   265,
     266,   262,   429,   762,   763,   263,   264,   265,   266,   262,
     446,   764,   765,   263,   264,   265,   266,   766,   452,   767,
     262,   768,   787,   769,   263,   264,   265,   266,   262,   454,
     770,   771,   263,   264,   265,   266,   788,   456,   789,   794,
     262,   791,   795,   796,   263,   264,   265,   266,   262,   458,
     797,   798,   263,   264,   265,   266,   262,   460,   799,     0,
     263,   264,   265,   266,     0,   462,     0,   262,     0,     0,
       0,   263,   264,   265,   266,   262,   464,     0,     0,   263,
     264,   265,   266,     0,   466,     0,     0,   262,     0,     0,
       0,   263,   264,   265,   266,   262,   475,     0,     0,   263,
     264,   265,   266,   262,   477,     0,     0,   263,   264,   265,
     266,     0,   479,     0,   262,     0,     0,     0,   263,   264,
     265,   266,   262,     0,     0,     0,   263,   264,   265,   266,
       0,     0,     0,     0,   262,     0,     0,     0,   263,   264,
     265,   266,   262,     0,     0,     0,   263,   264,   265,   266,
     262,     0,     0,     0,   263,   264,   265,   266,   662,   664,
     666,   668,   670,   672,   674,   676,   678,   680,   262,     0,
       0,   727,   263,   264,   265,   266,   262,     0,     0,   729,
     263,   264,   265,   266,   262,     0,     0,   731,   263,   264,
     265,   266
};

static const yytype_int16 yycheck[] =
{
      25,    26,   246,   247,   246,   247,     1,    39,     4,   253,
      42,   253,    37,    38,     4,     4,     1,     1,     1,     4,
      45,    46,    47,    48,    49,    50,    51,     1,     3,     1,
       4,     1,     4,     1,    39,     3,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    42,   248,   249,
     250,    76,   252,    42,     1,   255,    39,    42,    42,   259,
      43,    44,    45,    46,     4,     1,    91,     1,    42,    39,
      42,    79,    80,    43,    44,    45,    46,   102,    48,    49,
       1,    39,    39,   108,    42,    42,   111,   112,   113,   114,
     115,   116,   117,   118,     1,    42,     4,    42,   123,   124,
     125,    40,    41,    42,    43,    44,    42,     1,    42,     3,
     135,   136,     1,    52,    53,     4,    55,     4,    39,    58,
      59,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,     1,    39,    54,    42,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    84,    85,    86,    87,    88,
      89,    90,     4,    42,    39,     1,     4,    42,     4,     1,
      39,     4,   101,    42,    43,    44,    45,    46,     4,    39,
     109,   110,     1,    43,    44,    45,    46,     4,    48,    49,
     119,   120,   121,    39,    39,     1,    42,    42,    48,   128,
     129,   130,   131,   132,   133,   134,    42,    39,   137,   138,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,    54,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    39,     1,    54,    42,
      43,    44,    45,    46,     4,    48,    49,     4,     1,    39,
       1,     4,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     4,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     4,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,    42,
      54,     4,    43,    44,    45,    46,     1,     1,     3,     3,
     544,     1,     4,     0,     1,     1,     3,     5,     6,     4,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
     520,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    42,    39,    41,   545,    39,    43,    44,    45,
      46,    39,    39,    39,    48,    52,    53,     6,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,     4,    39,
       1,     1,    42,     4,     4,     1,     3,    47,    48,    49,
      50,    51,    39,     3,    54,    42,     1,    42,     1,     4,
      47,    48,    49,    50,    51,    40,    39,    54,    40,     5,
       5,     1,     1,    39,     4,    40,     4,    43,    44,    45,
      46,    42,    42,    39,     1,   749,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    39,    42,    54,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,     1,
      39,    54,    42,    42,   559,   560,   561,   562,   563,   564,
     565,   566,   567,   568,     5,    42,     1,     1,     1,   557,
      40,     1,     4,    40,     4,     1,     5,     5,     5,   584,
     585,   586,   587,   588,   589,   573,     5,    39,     5,     5,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
     605,   606,   607,   608,     4,    39,    39,    42,    42,    42,
      42,   599,    42,    39,   602,   603,   604,    43,    44,    45,
      46,   609,   610,   611,     1,     5,     1,   546,   547,   548,
     549,   550,   551,   552,   553,   554,   555,   556,    39,    39,
       5,    42,     1,    43,    44,    45,    46,    48,    49,     5,
     569,   570,     5,    54,     5,   574,   575,   576,   577,   578,
     579,   580,   581,   582,   583,    42,     1,    42,     5,     5,
      39,   590,   591,   592,    43,    44,    45,    46,     5,     1,
      39,   600,   601,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,     1,     5,    54,     5,   616,   617,   618,
     619,   620,   621,   622,    39,     5,     1,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    39,     5,    54,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
       1,    39,    54,     5,     3,    43,    44,    45,    46,     5,
       5,     5,     5,     1,    39,     5,   751,   752,    43,    44,
      45,    46,   757,   758,     5,     5,   761,   762,     5,     1,
       5,     5,     5,     1,     5,     5,     5,    40,    39,     5,
       5,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,     5,     5,    54,    39,     5,     5,
      42,    39,     5,     3,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,     5,     5,    54,     5,     5,     5,
      39,     5,     5,     5,    43,    44,    45,    46,     5,     5,
       5,     5,     5,     5,   753,   754,   755,   756,     5,     5,
     759,   760,     5,     5,   763,   764,     5,     5,     5,     5,
     769,   770,    52,    53,     5,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
     120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   137,   138,   139,
     140,   141,   142,   143,   144,   145,   146,   147,   148,   149,
     150,   151,   152,   153,   154,     1,     5,    42,     5,    40,
       5,     5,    47,    48,    49,    50,    51,    39,     1,    54,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,     5,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    39,     1,    54,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    39,
       1,    54,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     1,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     1,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,     1,
      54,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    39,     1,    54,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    39,     1,    54,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,    54,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    39,     1,    54,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    39,
       1,    54,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     1,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     1,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,     1,
      54,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    39,     1,    54,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    39,     1,    54,
       5,    43,    44,    45,    46,     5,     1,     5,     5,     5,
      39,     5,     5,     5,    43,    44,    45,    46,     1,     5,
       5,     5,     4,    39,     5,     5,     1,    43,    44,    45,
      46,     5,     4,    42,     1,     4,    39,    39,     4,     4,
      43,    44,    45,    46,    39,     1,     4,   136,    43,    44,
      45,    46,    39,     1,    42,     5,    39,   749,     5,     5,
      43,    44,    45,    46,    39,     1,    39,     5,    43,    44,
      45,    46,    39,     1,     5,    39,    43,    44,    45,    46,
      39,     1,    39,    39,     5,    40,    40,    43,    44,    45,
      46,    39,     1,     5,     5,    43,    44,    45,    46,     5,
       1,     5,     5,    39,   141,     5,     5,    43,    44,    45,
      46,    39,     1,     5,     5,    43,    44,    45,    46,    39,
       1,     5,     5,    43,    44,    45,    46,     5,     1,     5,
      39,     5,    42,     5,    43,    44,    45,    46,    39,     1,
       5,     4,    43,    44,    45,    46,    42,     1,    42,     5,
      39,   768,     5,     4,    43,    44,    45,    46,    39,     1,
      42,     5,    43,    44,    45,    46,    39,     1,     4,    -1,
      43,    44,    45,    46,    -1,     1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,     1,    -1,    -1,    43,
      44,    45,    46,    -1,     1,    -1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,     1,    -1,    -1,    43,
      44,    45,    46,    39,     1,    -1,    -1,    43,    44,    45,
      46,    -1,     1,    -1,    39,    -1,    -1,    -1,    43,    44,
      45,    46,    39,    -1,    -1,    -1,    43,    44,    45,    46,
      -1,    -1,    -1,    -1,    39,    -1,    -1,    -1,    43,    44,
      45,    46,    39,    -1,    -1,    -1,    43,    44,    45,    46,
      39,    -1,    -1,    -1,    43,    44,    45,    46,   559,   560,
     561,   562,   563,   564,   565,   566,   567,   568,    39,    -1,
      -1,    42,    43,    44,    45,    46,    39,    -1,    -1,    42,
      43,    44,    45,    46,    39,    -1,    -1,    42,    43,    44,
      45,    46
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   156,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    41,    52,    53,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   157,
     166,   170,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     3,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,    39,     4,     4,
       4,     4,    39,    43,    44,    45,    46,   160,   161,   160,
      39,    42,    48,    49,    54,    42,    48,    48,    39,   168,
      39,   169,    39,    39,     6,     1,   160,     1,   160,     1,
      42,    47,    48,    49,    50,    51,    54,   158,   159,   160,
     163,   164,     1,   158,     1,   158,     1,   158,     1,   158,
       1,   158,     1,   160,     1,   160,     1,   160,     1,   160,
       1,   160,     1,   160,     1,   160,     1,   158,     1,   158,
       1,     1,   158,     1,     1,    42,     1,   158,     1,   158,
       1,   160,     1,   160,     1,   160,     1,   160,     1,   160,
       1,   160,     1,   160,     1,   160,     1,   160,     1,   160,
       1,   160,     1,     4,    42,     1,     4,    42,     1,     4,
      42,     1,    42,     1,   160,     1,    42,     1,    42,     1,
     159,     1,   159,     1,     4,    42,     1,     4,    42,     1,
       4,    42,     1,   158,     1,   158,     1,   158,     1,   158,
       1,   158,     1,   158,     1,   158,     1,   160,     1,    42,
       1,    42,     1,    42,     1,    42,     1,    42,     1,     1,
       4,    42,     1,     4,    42,     1,    42,     1,   158,     1,
     160,     1,     4,    42,     1,    39,    42,     1,    39,    42,
       1,    39,    42,     1,    39,    42,     1,   160,     1,   158,
       1,   158,     1,   160,     1,   160,     1,   160,     1,   160,
       1,   160,     1,   160,     1,   160,     1,   160,     1,   158,
       1,   158,     1,   158,     1,     1,   160,     1,   160,     1,
     160,     1,     4,    42,     1,     4,    42,     1,   158,     1,
     158,     1,   158,     1,   158,     1,   158,     1,   158,     1,
     158,     1,    48,    49,   160,   165,     1,   160,   165,     1,
     158,     1,   158,     3,     3,   174,     3,    42,   173,    54,
     162,   163,   162,   173,   173,   173,    40,   173,    39,   162,
      39,   173,    40,   173,     5,     5,     5,     5,     6,     5,
       5,     5,     5,     5,   167,     4,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     3,    40,   173,    40,    40,    40,    40,
       4,    42,     4,    42,     4,     4,    42,     4,     4,     4,
      39,    39,    39,   162,   171,   172,   173,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   159,     4,
      42,   160,   164,   160,   164,   160,   164,   160,   164,   160,
     164,   160,   164,   160,   164,   160,   164,   160,   164,   160,
     164,   158,   158,     4,    42,   159,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   160,   160,     4,   160,
       4,   160,   160,   160,   158,   158,   158,    39,    42,    39,
      42,    39,    42,    39,    42,    39,    42,    39,    42,   159,
     158,   158,   159,   159,   159,    42,   160,    42,   160,    42,
     160,    42,   160,   159,   159,   159,    39,    39,    39,    39,
     158,   158,   158,   158,   158,   158,   158,    40,     5,     5,
      40,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     4,   172,   160,   160,   158,   158,   158,   158,   160,
     160,   158,   158,   160,   160,   158,   158,    42,    42,    42,
     241,   241,   158,   158,     5,     5,     4,    42,     5,     4
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
#line 242 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); ;}
    break;

  case 31:
#line 256 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); ;}
    break;

  case 32:
#line 257 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 33:
#line 258 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 34:
#line 259 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 35:
#line 260 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 36:
#line 261 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 37:
#line 262 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 38:
#line 263 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 39:
#line 264 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 40:
#line 265 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 41:
#line 266 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 42:
#line 267 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 43:
#line 268 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 44:
#line 269 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 45:
#line 270 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 46:
#line 271 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 47:
#line 272 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); ;}
    break;

  case 48:
#line 273 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 49:
#line 274 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 50:
#line 275 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 51:
#line 276 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 52:
#line 277 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 53:
#line 278 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 54:
#line 279 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 55:
#line 280 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 56:
#line 281 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 57:
#line 282 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 58:
#line 283 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); ;}
    break;

  case 59:
#line 284 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); ;}
    break;

  case 60:
#line 285 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 61:
#line 286 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 62:
#line 287 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 65:
#line 290 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 66:
#line 291 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 67:
#line 292 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 68:
#line 293 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 69:
#line 294 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 70:
#line 295 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 71:
#line 296 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 72:
#line 297 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ ;}
    break;

  case 73:
#line 298 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); ;}
    break;

  case 75:
#line 299 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 76:
#line 300 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 77:
#line 301 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 78:
#line 303 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 79:
#line 308 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 80:
#line 316 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 81:
#line 317 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 82:
#line 321 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 83:
#line 322 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 84:
#line 325 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); ;}
    break;

  case 86:
#line 329 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 87:
#line 330 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 90:
#line 339 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); ;}
    break;

  case 193:
#line 448 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 194:
#line 449 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); ;}
    break;

  case 195:
#line 453 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 196:
#line 454 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); ;}
    break;

  case 197:
#line 458 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 198:
#line 459 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); ;}
    break;

  case 199:
#line 463 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 200:
#line 464 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); ;}
    break;

  case 201:
#line 468 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 202:
#line 469 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); ;}
    break;

  case 203:
#line 474 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 204:
#line 475 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); ;}
    break;

  case 205:
#line 479 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 206:
#line 480 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); ;}
    break;

  case 207:
#line 484 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 208:
#line 485 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); ;}
    break;

  case 209:
#line 489 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 210:
#line 490 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); ;}
    break;

  case 211:
#line 495 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 212:
#line 496 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); ;}
    break;

  case 213:
#line 500 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 214:
#line 501 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); ;}
    break;

  case 215:
#line 505 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 216:
#line 506 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); ;}
    break;

  case 217:
#line 510 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 218:
#line 511 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); ;}
    break;

  case 219:
#line 516 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 220:
#line 517 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); ;}
    break;

  case 221:
#line 521 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 222:
#line 522 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); ;}
    break;

  case 223:
#line 526 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 224:
#line 527 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); ;}
    break;

  case 225:
#line 531 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 226:
#line 532 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); ;}
    break;

  case 227:
#line 536 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 228:
#line 537 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); ;}
    break;

  case 229:
#line 541 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 230:
#line 542 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); ;}
    break;

  case 231:
#line 546 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 232:
#line 547 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 233:
#line 548 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); ;}
    break;

  case 234:
#line 552 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 235:
#line 553 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); ;}
    break;

  case 236:
#line 557 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 237:
#line 558 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); ;}
    break;

  case 238:
#line 562 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 239:
#line 563 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); ;}
    break;

  case 240:
#line 567 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 241:
#line 568 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); ;}
    break;

  case 242:
#line 572 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 243:
#line 573 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 244:
#line 574 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); ;}
    break;

  case 245:
#line 578 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 246:
#line 579 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 247:
#line 580 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); ;}
    break;

  case 248:
#line 584 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 249:
#line 585 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); ;}
    break;

  case 250:
#line 589 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 251:
#line 590 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); ;}
    break;

  case 252:
#line 595 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 253:
#line 596 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); ;}
    break;

  case 254:
#line 597 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); ;}
    break;

  case 255:
#line 601 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 256:
#line 602 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); ;}
    break;

  case 257:
#line 607 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 258:
#line 608 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); ;}
    break;

  case 259:
#line 612 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 260:
#line 613 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); ;}
    break;

  case 261:
#line 617 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 262:
#line 618 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); ;}
    break;

  case 263:
#line 623 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 264:
#line 624 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 265:
#line 625 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); ;}
    break;

  case 266:
#line 629 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 267:
#line 630 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 268:
#line 631 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); ;}
    break;

  case 269:
#line 635 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 270:
#line 636 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 271:
#line 637 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); ;}
    break;

  case 272:
#line 641 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 273:
#line 642 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 274:
#line 643 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); ;}
    break;

  case 275:
#line 647 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 276:
#line 648 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 277:
#line 649 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); ;}
    break;

  case 278:
#line 653 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 279:
#line 654 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 280:
#line 655 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; ;}
    break;

  case 281:
#line 659 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 282:
#line 660 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 283:
#line 661 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; ;}
    break;

  case 284:
#line 665 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 285:
#line 666 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 286:
#line 667 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); ;}
    break;

  case 287:
#line 671 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 288:
#line 672 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 289:
#line 673 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); ;}
    break;

  case 290:
#line 677 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 291:
#line 678 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 292:
#line 679 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); ;}
    break;

  case 293:
#line 683 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 294:
#line 684 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 295:
#line 685 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); ;}
    break;

  case 296:
#line 689 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 297:
#line 690 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 298:
#line 691 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); ;}
    break;

  case 299:
#line 695 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 300:
#line 696 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 301:
#line 697 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); ;}
    break;

  case 302:
#line 701 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 303:
#line 702 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); ;}
    break;

  case 304:
#line 706 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 305:
#line 707 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); ;}
    break;

  case 306:
#line 711 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 307:
#line 712 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); ;}
    break;

  case 308:
#line 716 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 309:
#line 717 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); ;}
    break;

  case 310:
#line 721 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 311:
#line 722 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); ;}
    break;

  case 312:
#line 726 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 313:
#line 727 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); ;}
    break;

  case 314:
#line 731 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 315:
#line 732 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 316:
#line 733 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); ;}
    break;

  case 317:
#line 737 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); ;}
    break;

  case 318:
#line 738 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); ;}
    break;

  case 319:
#line 742 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 320:
#line 743 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 321:
#line 744 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); ;}
    break;

  case 322:
#line 748 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 323:
#line 749 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 324:
#line 750 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); ;}
    break;

  case 325:
#line 755 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 326:
#line 756 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 327:
#line 757 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); ;}
    break;

  case 328:
#line 761 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 329:
#line 762 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 330:
#line 763 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); ;}
    break;

  case 331:
#line 767 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); ;}
    break;

  case 332:
#line 768 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); ;}
    break;

  case 333:
#line 772 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); ;}
    break;

  case 334:
#line 773 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); ;}
    break;

  case 335:
#line 777 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 336:
#line 778 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); ;}
    break;

  case 337:
#line 782 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); ;}
    break;

  case 338:
#line 783 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); ;}
    break;

  case 339:
#line 787 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 340:
#line 788 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); ;}
    break;

  case 341:
#line 792 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); ;}
    break;

  case 342:
#line 793 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); ;}
    break;

  case 343:
#line 797 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 344:
#line 798 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); ;}
    break;

  case 345:
#line 802 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 346:
#line 803 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); ;}
    break;

  case 347:
#line 808 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      ;}
    break;

  case 348:
#line 817 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      ;}
    break;

  case 349:
#line 825 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 350:
#line 826 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 351:
#line 827 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); ;}
    break;

  case 352:
#line 831 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 353:
#line 832 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 354:
#line 833 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 355:
#line 834 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 356:
#line 835 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); ;}
    break;

  case 357:
#line 839 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 358:
#line 840 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 359:
#line 841 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 360:
#line 842 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 361:
#line 843 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); ;}
    break;

  case 362:
#line 847 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 363:
#line 848 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 364:
#line 849 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 365:
#line 850 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 366:
#line 851 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 367:
#line 855 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 368:
#line 856 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 369:
#line 857 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 370:
#line 861 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 371:
#line 862 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); ;}
    break;

  case 372:
#line 866 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 373:
#line 867 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); ;}
    break;

  case 374:
#line 871 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 375:
#line 872 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); ;}
    break;

  case 376:
#line 876 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 377:
#line 877 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); ;}
    break;

  case 378:
#line 881 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 379:
#line 882 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); ;}
    break;

  case 380:
#line 886 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 381:
#line 887 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); ;}
    break;

  case 382:
#line 891 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 383:
#line 892 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); ;}
    break;

  case 384:
#line 896 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 385:
#line 897 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); ;}
    break;

  case 386:
#line 901 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 387:
#line 902 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 388:
#line 903 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); ;}
    break;

  case 389:
#line 907 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 390:
#line 908 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 391:
#line 909 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); ;}
    break;

  case 392:
#line 913 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 393:
#line 914 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 394:
#line 915 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); ;}
    break;

  case 395:
#line 919 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 396:
#line 920 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 397:
#line 921 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); ;}
    break;

  case 398:
#line 926 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 399:
#line 927 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); ;}
    break;

  case 400:
#line 931 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 401:
#line 932 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); ;}
    break;

  case 402:
#line 936 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 403:
#line 937 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); ;}
    break;

  case 404:
#line 941 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 405:
#line 942 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); ;}
    break;

  case 406:
#line 946 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 407:
#line 947 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); ;}
    break;

  case 408:
#line 951 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_FORI, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 409:
#line 952 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_FORI, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 410:
#line 953 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORI" ); ;}
    break;

  case 411:
#line 957 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_FORN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 412:
#line 958 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_FORN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 413:
#line 959 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORN" ); ;}
    break;

  case 414:
#line 963 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 415:
#line 964 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); ;}
    break;

  case 416:
#line 968 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 417:
#line 969 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); ;}
    break;

  case 418:
#line 973 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 419:
#line 974 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); ;}
    break;

  case 420:
#line 978 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 421:
#line 979 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); ;}
    break;

  case 422:
#line 983 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 423:
#line 984 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); ;}
    break;

  case 424:
#line 988 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 425:
#line 989 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); ;}
    break;

  case 426:
#line 993 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 427:
#line 994 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); ;}
    break;

  case 428:
#line 998 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 429:
#line 999 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 430:
#line 1000 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); ;}
    break;

  case 431:
#line 1004 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 432:
#line 1005 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 433:
#line 1006 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); ;}
    break;

  case 434:
#line 1010 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 435:
#line 1011 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); ;}
    break;

  case 436:
#line 1015 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 437:
#line 1016 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 4110 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"
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


#line 1019 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


