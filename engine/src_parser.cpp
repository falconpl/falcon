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
#define yyparse flc_src_parse
#define yylex   flc_src_lex
#define yyerror flc_src_error
#define yylval  flc_src_lval
#define yychar  flc_src_char
#define yydebug flc_src_debug
#define yynerrs flc_src_nerrs


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




/* Copy the first part of user declarations.  */
#line 22 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->ctxOpenLine() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
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
/* Line 187 of yacc.c.  */
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 391 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6066

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  110
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  158
/* YYNRULES -- Number of rules.  */
#define YYNRULES  425
/* YYNRULES -- Number of states.  */
#define YYNSTATES  773

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   364

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
     105,   106,   107,   108,   109
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    63,    67,    71,    74,
      77,    82,    89,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   129,   135,   139,   143,   144,   150,   153,   157,   159,
     163,   167,   170,   174,   175,   182,   185,   189,   193,   197,
     201,   202,   204,   205,   209,   212,   216,   217,   222,   226,
     230,   231,   234,   237,   241,   244,   248,   252,   253,   263,
     264,   272,   278,   282,   283,   286,   288,   290,   292,   294,
     298,   302,   306,   309,   313,   316,   320,   324,   326,   327,
     334,   338,   342,   343,   350,   354,   358,   359,   366,   370,
     374,   375,   382,   386,   390,   391,   394,   398,   400,   401,
     407,   408,   414,   415,   421,   422,   428,   429,   430,   434,
     435,   437,   440,   443,   446,   448,   452,   454,   456,   458,
     462,   464,   465,   472,   476,   480,   481,   484,   488,   490,
     491,   497,   498,   504,   505,   511,   512,   518,   520,   524,
     525,   527,   529,   535,   540,   544,   548,   549,   556,   559,
     563,   564,   566,   568,   571,   574,   577,   582,   586,   592,
     596,   598,   602,   604,   606,   610,   614,   620,   623,   629,
     630,   638,   642,   648,   649,   656,   659,   660,   662,   666,
     668,   669,   670,   676,   677,   681,   684,   688,   691,   695,
     699,   703,   707,   713,   719,   723,   729,   735,   739,   742,
     746,   750,   752,   756,   760,   764,   766,   770,   774,   778,
     780,   784,   788,   792,   797,   801,   804,   808,   811,   815,
     816,   818,   822,   825,   829,   832,   833,   842,   846,   849,
     850,   854,   855,   861,   862,   865,   867,   871,   874,   875,
     879,   881,   885,   887,   889,   891,   892,   895,   897,   899,
     901,   903,   904,   912,   918,   923,   924,   928,   932,   934,
     937,   941,   946,   947,   956,   959,   962,   963,   966,   968,
     970,   972,   974,   975,   980,   982,   986,   990,   992,   995,
     999,  1003,  1005,  1007,  1009,  1011,  1013,  1015,  1017,  1019,
    1021,  1023,  1025,  1027,  1030,  1034,  1038,  1042,  1046,  1050,
    1054,  1058,  1062,  1066,  1070,  1074,  1077,  1081,  1084,  1087,
    1090,  1093,  1097,  1101,  1105,  1109,  1113,  1117,  1121,  1124,
    1128,  1132,  1136,  1140,  1144,  1147,  1150,  1153,  1156,  1158,
    1160,  1162,  1164,  1166,  1168,  1171,  1173,  1178,  1184,  1188,
    1190,  1192,  1196,  1202,  1206,  1210,  1214,  1218,  1222,  1226,
    1230,  1234,  1238,  1242,  1246,  1250,  1254,  1259,  1264,  1270,
    1278,  1283,  1287,  1288,  1295,  1296,  1303,  1308,  1312,  1315,
    1316,  1323,  1324,  1330,  1332,  1335,  1341,  1347,  1352,  1356,
    1359,  1363,  1367,  1370,  1374,  1378,  1382,  1386,  1391,  1393,
    1397,  1399,  1402,  1404,  1408,  1412
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     111,     0,    -1,   112,    -1,    -1,   112,   113,    -1,   114,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   116,    -1,
     209,    -1,   189,    -1,   217,    -1,   235,    -1,   117,    -1,
     204,    -1,   205,    -1,   207,    -1,   212,    -1,     4,    -1,
      97,     4,    -1,    39,     6,     3,    -1,    39,     7,     3,
      -1,    39,     1,     3,    -1,   118,    -1,     3,    -1,    48,
       1,     3,    -1,    34,     1,     3,    -1,    32,     1,     3,
      -1,     1,     3,    -1,   248,     3,    -1,   264,    74,   248,
       3,    -1,   264,    74,   248,    76,   264,     3,    -1,   120,
      -1,   121,    -1,   138,    -1,   152,    -1,   167,    -1,   125,
      -1,   136,    -1,   137,    -1,   178,    -1,   179,    -1,   188,
      -1,   244,    -1,   240,    -1,   202,    -1,   203,    -1,   143,
      -1,   144,    -1,   145,    -1,   246,    74,   248,    -1,   119,
      76,   246,    74,   248,    -1,    10,   119,     3,    -1,    10,
       1,     3,    -1,    -1,   123,   122,   135,     9,     3,    -1,
     124,   117,    -1,    11,   248,     3,    -1,    53,    -1,    11,
       1,     3,    -1,    11,   248,    47,    -1,    53,    47,    -1,
      11,     1,    47,    -1,    -1,   127,   126,   135,   129,     9,
       3,    -1,   128,   117,    -1,    15,   248,     3,    -1,    15,
       1,     3,    -1,    15,   248,    47,    -1,    15,     1,    47,
      -1,    -1,   132,    -1,    -1,   131,   130,   135,    -1,    16,
       3,    -1,    16,     1,     3,    -1,    -1,   134,   133,   135,
     129,    -1,    17,   248,     3,    -1,    17,     1,     3,    -1,
      -1,   135,   117,    -1,    12,     3,    -1,    12,     1,     3,
      -1,    13,     3,    -1,    13,    14,     3,    -1,    13,     1,
       3,    -1,    -1,    18,   266,    89,   248,     3,   139,   141,
       9,     3,    -1,    -1,    18,   266,    89,   248,    47,   140,
     117,    -1,    18,   266,    89,     1,     3,    -1,    18,     1,
       3,    -1,    -1,   142,   141,    -1,   117,    -1,   146,    -1,
     148,    -1,   150,    -1,    51,   248,     3,    -1,    51,     1,
       3,    -1,   103,   264,     3,    -1,   103,     3,    -1,    84,
     264,     3,    -1,    84,     3,    -1,   103,     1,     3,    -1,
      84,     1,     3,    -1,    56,    -1,    -1,    19,     3,   147,
     135,     9,     3,    -1,    19,    47,   117,    -1,    19,     1,
       3,    -1,    -1,    20,     3,   149,   135,     9,     3,    -1,
      20,    47,   117,    -1,    20,     1,     3,    -1,    -1,    21,
       3,   151,   135,     9,     3,    -1,    21,    47,   117,    -1,
      21,     1,     3,    -1,    -1,   154,   153,   155,   161,     9,
       3,    -1,    22,   248,     3,    -1,    22,     1,     3,    -1,
      -1,   155,   156,    -1,   155,     1,     3,    -1,     3,    -1,
      -1,    23,   165,     3,   157,   135,    -1,    -1,    23,   165,
      47,   158,   117,    -1,    -1,    23,     1,     3,   159,   135,
      -1,    -1,    23,     1,    47,   160,   117,    -1,    -1,    -1,
     163,   162,   164,    -1,    -1,    24,    -1,    24,     1,    -1,
       3,   135,    -1,    47,   117,    -1,   166,    -1,   165,    76,
     166,    -1,     8,    -1,   115,    -1,     7,    -1,   115,    75,
     115,    -1,     6,    -1,    -1,   169,   168,   170,   161,     9,
       3,    -1,    25,   248,     3,    -1,    25,     1,     3,    -1,
      -1,   170,   171,    -1,   170,     1,     3,    -1,     3,    -1,
      -1,    23,   176,     3,   172,   135,    -1,    -1,    23,   176,
      47,   173,   117,    -1,    -1,    23,     1,     3,   174,   135,
      -1,    -1,    23,     1,    47,   175,   117,    -1,   177,    -1,
     176,    76,   177,    -1,    -1,     4,    -1,     6,    -1,    28,
     264,    75,   248,     3,    -1,    28,   264,     1,     3,    -1,
      28,     1,     3,    -1,    29,    47,   117,    -1,    -1,   181,
     180,   135,   182,     9,     3,    -1,    29,     3,    -1,    29,
       1,     3,    -1,    -1,   183,    -1,   184,    -1,   183,   184,
      -1,   185,   135,    -1,    30,     3,    -1,    30,    89,   246,
       3,    -1,    30,   186,     3,    -1,    30,   186,    89,   246,
       3,    -1,    30,     1,     3,    -1,   187,    -1,   186,    76,
     187,    -1,     4,    -1,     6,    -1,    31,   248,     3,    -1,
      31,     1,     3,    -1,   190,   197,   135,     9,     3,    -1,
     192,   117,    -1,   194,    58,   195,    57,     3,    -1,    -1,
     194,    58,   195,     1,   191,    57,     3,    -1,   194,     1,
       3,    -1,   194,    58,   195,    57,    47,    -1,    -1,   194,
      58,     1,   193,    57,    47,    -1,    48,     6,    -1,    -1,
     196,    -1,   195,    76,   196,    -1,     6,    -1,    -1,    -1,
     200,   198,   135,     9,     3,    -1,    -1,   201,   199,   117,
      -1,    49,     3,    -1,    49,     1,     3,    -1,    49,    47,
      -1,    49,     1,    47,    -1,    40,   250,     3,    -1,    40,
       1,     3,    -1,    43,   248,     3,    -1,    43,   248,    89,
     248,     3,    -1,    43,   248,    89,     1,     3,    -1,    43,
       1,     3,    -1,    41,     6,    74,   245,     3,    -1,    41,
       6,    74,     1,     3,    -1,    41,     1,     3,    -1,    44,
       3,    -1,    44,   206,     3,    -1,    44,     1,     3,    -1,
       6,    -1,   206,    76,     6,    -1,    45,   208,     3,    -1,
      45,     1,     3,    -1,     6,    -1,   206,    76,     6,    -1,
      46,   210,     3,    -1,    46,     1,     3,    -1,   211,    -1,
     210,    76,   211,    -1,     6,    74,     6,    -1,     6,    74,
     115,    -1,   213,   216,     9,     3,    -1,   214,   215,     3,
      -1,    42,     3,    -1,    42,     1,     3,    -1,    42,    47,
      -1,    42,     1,    47,    -1,    -1,     6,    -1,   215,    76,
       6,    -1,   215,     3,    -1,   216,   215,     3,    -1,     1,
       3,    -1,    -1,    32,     6,   218,   219,   228,   233,     9,
       3,    -1,   220,   222,     3,    -1,     1,     3,    -1,    -1,
      58,   195,    57,    -1,    -1,    58,   195,     1,   221,    57,
      -1,    -1,    33,   223,    -1,   224,    -1,   223,    76,   224,
      -1,     6,   225,    -1,    -1,    58,   226,    57,    -1,   227,
      -1,   226,    76,   227,    -1,   245,    -1,     6,    -1,    27,
      -1,    -1,   228,   229,    -1,     3,    -1,   189,    -1,   232,
      -1,   230,    -1,    -1,    38,     3,   231,   197,   135,     9,
       3,    -1,    49,     6,    74,   248,     3,    -1,     6,    74,
     248,     3,    -1,    -1,    91,   234,     3,    -1,    91,     1,
       3,    -1,     6,    -1,    80,     6,    -1,   234,    76,     6,
      -1,   234,    76,    80,     6,    -1,    -1,    34,     6,   236,
     237,   238,   233,     9,     3,    -1,   222,     3,    -1,     1,
       3,    -1,    -1,   238,   239,    -1,     3,    -1,   189,    -1,
     232,    -1,   230,    -1,    -1,    36,   241,   242,     3,    -1,
     243,    -1,   242,    76,   243,    -1,   242,    76,     1,    -1,
       6,    -1,    35,     3,    -1,    35,   248,     3,    -1,    35,
       1,     3,    -1,     8,    -1,    54,    -1,    55,    -1,     4,
      -1,     5,    -1,     7,    -1,     6,    -1,   246,    -1,    27,
      -1,    26,    -1,   245,    -1,   247,    -1,    97,   248,    -1,
     248,    98,   248,    -1,   248,    97,   248,    -1,   248,   101,
     248,    -1,   248,   100,   248,    -1,   248,    99,   248,    -1,
     248,   102,   248,    -1,   248,    96,   248,    -1,   248,    95,
     248,    -1,   248,    94,   248,    -1,   248,   104,   248,    -1,
     248,   103,   248,    -1,   105,   248,    -1,   248,    85,   248,
      -1,   248,   108,    -1,   108,   248,    -1,   248,   107,    -1,
     107,   248,    -1,   248,    86,   248,    -1,   248,    84,   248,
      -1,   248,    83,   248,    -1,   248,    82,   248,    -1,   248,
      81,   248,    -1,   248,    79,   248,    -1,   248,    78,   248,
      -1,    80,   248,    -1,   248,    91,   248,    -1,   248,    90,
     248,    -1,   248,    89,   248,    -1,   248,    88,   248,    -1,
     248,    87,     6,    -1,   109,   246,    -1,   109,   109,    -1,
      93,   248,    -1,    92,   248,    -1,   257,    -1,   252,    -1,
     255,    -1,   250,    -1,   260,    -1,   262,    -1,   248,   249,
      -1,   261,    -1,   248,    60,   248,    59,    -1,   248,    60,
     101,   248,    59,    -1,   248,    61,     6,    -1,   263,    -1,
     249,    -1,   248,    74,   248,    -1,   248,    74,   248,    76,
     264,    -1,   248,    73,   248,    -1,   248,    72,   248,    -1,
     248,    71,   248,    -1,   248,    70,   248,    -1,   248,    69,
     248,    -1,   248,    63,   248,    -1,   248,    68,   248,    -1,
     248,    67,   248,    -1,   248,    66,   248,    -1,   248,    64,
     248,    -1,   248,    65,   248,    -1,    58,   248,    57,    -1,
      60,    47,    59,    -1,    60,   248,    47,    59,    -1,    60,
      47,   248,    59,    -1,    60,   248,    47,   248,    59,    -1,
      60,   248,    47,   248,    47,   248,    59,    -1,   248,    58,
     264,    57,    -1,   248,    58,    57,    -1,    -1,   248,    58,
     264,     1,   251,    57,    -1,    -1,    48,   253,   254,   197,
     135,     9,    -1,    58,   195,    57,     3,    -1,    58,   195,
       1,    -1,     1,     3,    -1,    -1,    50,   256,   254,   197,
     135,     9,    -1,    -1,    37,   258,   259,    62,   248,    -1,
     195,    -1,     1,     3,    -1,   248,    77,   248,    47,   248,
      -1,   248,    77,   248,    47,     1,    -1,   248,    77,   248,
       1,    -1,   248,    77,     1,    -1,    60,    59,    -1,    60,
     264,    59,    -1,    60,   264,     1,    -1,    52,    59,    -1,
      52,   265,    59,    -1,    52,   265,     1,    -1,    60,    62,
      59,    -1,    60,   267,    59,    -1,    60,   267,     1,    59,
      -1,   248,    -1,   264,    76,   248,    -1,   248,    -1,   265,
     248,    -1,   246,    -1,   266,    76,   246,    -1,   248,    62,
     248,    -1,   267,    76,   248,    62,   248,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   199,   199,   202,   204,   208,   209,   210,   214,   219,
     220,   225,   230,   235,   240,   241,   242,   243,   247,   248,
     252,   258,   264,   272,   273,   274,   275,   276,   277,   282,
     284,   290,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,   317,   318,   319,   320,   321,   322,
     326,   332,   340,   342,   347,   347,   361,   369,   370,   371,
     375,   376,   377,   381,   381,   396,   406,   407,   411,   412,
     416,   418,   419,   419,   428,   429,   434,   434,   446,   447,
     450,   452,   458,   467,   475,   485,   494,   504,   503,   528,
     527,   553,   558,   565,   567,   571,   578,   579,   580,   584,
     597,   605,   609,   615,   621,   628,   633,   642,   652,   652,
     666,   675,   679,   679,   692,   701,   705,   705,   721,   730,
     734,   734,   751,   752,   759,   761,   762,   766,   768,   767,
     778,   778,   790,   790,   802,   802,   818,   821,   820,   833,
     834,   835,   838,   839,   845,   846,   850,   859,   871,   882,
     893,   914,   914,   931,   932,   939,   941,   942,   946,   948,
     947,   958,   958,   971,   971,   983,   983,  1001,  1002,  1005,
    1006,  1018,  1039,  1043,  1048,  1056,  1063,  1062,  1081,  1082,
    1085,  1087,  1091,  1092,  1096,  1101,  1119,  1139,  1149,  1160,
    1168,  1169,  1173,  1185,  1208,  1209,  1216,  1226,  1235,  1236,
    1236,  1240,  1244,  1245,  1245,  1252,  1306,  1308,  1309,  1313,
    1328,  1331,  1330,  1342,  1341,  1356,  1357,  1361,  1362,  1371,
    1375,  1383,  1390,  1405,  1411,  1423,  1433,  1438,  1450,  1459,
    1466,  1474,  1479,  1487,  1491,  1499,  1504,  1516,  1521,  1529,
    1530,  1534,  1538,  1550,  1557,  1567,  1568,  1571,  1572,  1575,
    1577,  1581,  1588,  1589,  1590,  1602,  1601,  1660,  1663,  1669,
    1671,  1672,  1672,  1678,  1680,  1684,  1685,  1689,  1723,  1725,
    1734,  1735,  1739,  1740,  1749,  1752,  1754,  1758,  1759,  1762,
    1780,  1784,  1784,  1818,  1840,  1867,  1869,  1870,  1877,  1885,
    1891,  1897,  1911,  1910,  1974,  1975,  1981,  1983,  1987,  1988,
    1991,  2010,  2019,  2018,  2036,  2037,  2038,  2045,  2061,  2062,
    2063,  2073,  2074,  2075,  2076,  2077,  2078,  2082,  2100,  2101,
    2102,  2113,  2114,  2115,  2116,  2117,  2118,  2119,  2120,  2121,
    2122,  2123,  2124,  2125,  2126,  2127,  2128,  2129,  2130,  2131,
    2132,  2133,  2134,  2135,  2136,  2137,  2138,  2139,  2140,  2141,
    2142,  2143,  2144,  2145,  2146,  2147,  2148,  2149,  2150,  2151,
    2152,  2153,  2154,  2155,  2157,  2162,  2166,  2171,  2177,  2186,
    2187,  2189,  2194,  2201,  2202,  2203,  2204,  2205,  2206,  2207,
    2208,  2209,  2210,  2211,  2212,  2217,  2220,  2223,  2226,  2229,
    2235,  2241,  2246,  2246,  2256,  2255,  2296,  2297,  2301,  2309,
    2308,  2354,  2353,  2396,  2397,  2406,  2411,  2418,  2425,  2435,
    2436,  2440,  2448,  2449,  2453,  2462,  2463,  2464,  2472,  2473,
    2477,  2478,  2482,  2488,  2495,  2496
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY",
  "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "LAMBDA", "INIT", "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS",
  "EXPORT", "IMPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC",
  "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "TRUE_TOKEN", "FALSE_TOKEN",
  "OUTER_STRING", "CLOSEPAR", "OPENPAR", "CLOSESQUARE", "OPENSQUARE",
  "DOT", "ARROW", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR",
  "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL",
  "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_TO", "COMMA", "QUESTION", "OR",
  "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ", "EEQ", "PROVIDES",
  "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN", "CAP", "VBAR",
  "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR", "POW", "SHR",
  "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR", "$accept",
  "input", "body", "line", "toplevel_statement", "INTNUM_WITH_MINUS",
  "load_statement", "statement", "base_statement", "assignment_def_list",
  "def_statement", "while_statement", "@1", "while_decl",
  "while_short_decl", "if_statement", "@2", "if_decl", "if_short_decl",
  "elif_or_else", "@3", "else_decl", "elif_statement", "@4", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "forin_statement", "@5", "@6", "forin_statement_list",
  "forin_statement_elem", "fordot_statement", "self_print_statement",
  "outer_print_statement", "first_loop_block", "@7", "last_loop_block",
  "@8", "middle_loop_block", "@9", "switch_statement", "@10",
  "switch_decl", "case_list", "case_statement", "@11", "@12", "@13", "@14",
  "default_statement", "@15", "default_decl", "default_body",
  "case_expression_list", "case_element", "select_statement", "@16",
  "select_decl", "selcase_list", "selcase_statement", "@17", "@18", "@19",
  "@20", "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@21", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@22", "func_decl_short", "@23", "func_begin", "param_list",
  "param_symbol", "static_block", "@24", "@25", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "import_statement", "import_symbol_list", "directive_statement",
  "directive_pair_list", "directive_pair", "attributes_statement",
  "attributes_decl", "attributes_short_decl", "attribute_list",
  "attribute_vert_list", "class_decl", "@26", "class_def_inner",
  "class_param_list", "@27", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@28", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@29", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@30", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "expression", "range_decl", "func_call", "@31",
  "nameless_func", "@32", "nameless_func_decl_inner", "nameless_closure",
  "@33", "lambda_expr", "@34", "lambda_expr_inner", "iif_expr",
  "array_decl", "dotarray_decl", "dict_decl", "expression_list",
  "listpar_expression_list", "symbol_list", "expression_pair_list", 0
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
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   110,   111,   112,   112,   113,   113,   113,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   115,   115,
     116,   116,   116,   117,   117,   117,   117,   117,   117,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
     119,   119,   120,   120,   122,   121,   121,   123,   123,   123,
     124,   124,   124,   126,   125,   125,   127,   127,   128,   128,
     129,   129,   130,   129,   131,   131,   133,   132,   134,   134,
     135,   135,   136,   136,   137,   137,   137,   139,   138,   140,
     138,   138,   138,   141,   141,   142,   142,   142,   142,   143,
     143,   144,   144,   144,   144,   144,   144,   145,   147,   146,
     146,   146,   149,   148,   148,   148,   151,   150,   150,   150,
     153,   152,   154,   154,   155,   155,   155,   156,   157,   156,
     158,   156,   159,   156,   160,   156,   161,   162,   161,   163,
     163,   163,   164,   164,   165,   165,   166,   166,   166,   166,
     166,   168,   167,   169,   169,   170,   170,   170,   171,   172,
     171,   173,   171,   174,   171,   175,   171,   176,   176,   177,
     177,   177,   178,   178,   178,   179,   180,   179,   181,   181,
     182,   182,   183,   183,   184,   185,   185,   185,   185,   185,
     186,   186,   187,   187,   188,   188,   189,   189,   190,   191,
     190,   190,   192,   193,   192,   194,   195,   195,   195,   196,
     197,   198,   197,   199,   197,   200,   200,   201,   201,   202,
     202,   203,   203,   203,   203,   204,   204,   204,   205,   205,
     205,   206,   206,   207,   207,   208,   208,   209,   209,   210,
     210,   211,   211,   212,   212,   213,   213,   214,   214,   215,
     215,   215,   216,   216,   216,   218,   217,   219,   219,   220,
     220,   221,   220,   222,   222,   223,   223,   224,   225,   225,
     226,   226,   227,   227,   227,   228,   228,   229,   229,   229,
     229,   231,   230,   232,   232,   233,   233,   233,   234,   234,
     234,   234,   236,   235,   237,   237,   238,   238,   239,   239,
     239,   239,   241,   240,   242,   242,   242,   243,   244,   244,
     244,   245,   245,   245,   245,   245,   245,   246,   247,   247,
     247,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   249,   249,   249,   249,   249,
     250,   250,   251,   250,   253,   252,   254,   254,   254,   256,
     255,   258,   257,   259,   259,   260,   260,   260,   260,   261,
     261,   261,   262,   262,   262,   263,   263,   263,   264,   264,
     265,   265,   266,   266,   267,   267
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     3,     3,     3,     2,     2,
       4,     6,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     5,     3,     3,     0,     5,     2,     3,     1,     3,
       3,     2,     3,     0,     6,     2,     3,     3,     3,     3,
       0,     1,     0,     3,     2,     3,     0,     4,     3,     3,
       0,     2,     2,     3,     2,     3,     3,     0,     9,     0,
       7,     5,     3,     0,     2,     1,     1,     1,     1,     3,
       3,     3,     2,     3,     2,     3,     3,     1,     0,     6,
       3,     3,     0,     6,     3,     3,     0,     6,     3,     3,
       0,     6,     3,     3,     0,     2,     3,     1,     0,     5,
       0,     5,     0,     5,     0,     5,     0,     0,     3,     0,
       1,     2,     2,     2,     1,     3,     1,     1,     1,     3,
       1,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     1,     3,     0,
       1,     1,     5,     4,     3,     3,     0,     6,     2,     3,
       0,     1,     1,     2,     2,     2,     4,     3,     5,     3,
       1,     3,     1,     1,     3,     3,     5,     2,     5,     0,
       7,     3,     5,     0,     6,     2,     0,     1,     3,     1,
       0,     0,     5,     0,     3,     2,     3,     2,     3,     3,
       3,     3,     5,     5,     3,     5,     5,     3,     2,     3,
       3,     1,     3,     3,     3,     1,     3,     3,     3,     1,
       3,     3,     3,     4,     3,     2,     3,     2,     3,     0,
       1,     3,     2,     3,     2,     0,     8,     3,     2,     0,
       3,     0,     5,     0,     2,     1,     3,     2,     0,     3,
       1,     3,     1,     1,     1,     0,     2,     1,     1,     1,
       1,     0,     7,     5,     4,     0,     3,     3,     1,     2,
       3,     4,     0,     8,     2,     2,     0,     2,     1,     1,
       1,     1,     0,     4,     1,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     2,     2,     2,
       2,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       3,     3,     3,     3,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     2,     1,     4,     5,     3,     1,
       1,     3,     5,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     4,     4,     5,     7,
       4,     3,     0,     6,     0,     6,     4,     3,     2,     0,
       6,     0,     5,     1,     2,     5,     5,     4,     3,     2,
       3,     3,     2,     3,     3,     3,     3,     4,     1,     3,
       1,     2,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   314,   315,   317,   316,
     311,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   320,   319,     0,     0,     0,     0,     0,     0,   302,
     401,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     399,     0,     0,    58,   312,   313,   107,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     4,
       5,     8,    13,    23,    32,    33,    54,     0,    37,    63,
       0,    38,    39,    34,    47,    48,    49,    35,   120,    36,
     151,    40,    41,   176,    42,    10,   210,     0,     0,    45,
      46,    14,    15,    16,     9,    17,     0,   249,    11,    12,
      44,    43,   321,   318,   322,   418,   370,   361,   359,   360,
     358,   362,   365,   363,   369,     0,    28,     6,     0,     0,
       0,     0,   394,     0,     0,    82,     0,    84,     0,     0,
       0,     0,   422,     0,     0,     0,     0,     0,     0,     0,
     418,     0,     0,   178,     0,     0,     0,     0,   255,     0,
     292,     0,   308,     0,     0,     0,     0,     0,     0,     0,
       0,   361,     0,     0,     0,   245,   247,     0,     0,     0,
     228,   231,     0,     0,   231,     0,     0,     0,     0,     0,
     239,     0,   205,     0,     0,     0,     0,   412,   420,     0,
      61,     0,     0,   409,     0,   418,     0,     0,   348,     0,
     104,     0,   357,   356,   323,     0,   102,     0,   335,   340,
     338,   355,   354,    80,     0,     0,     0,    56,    80,    65,
     124,   155,    80,     0,    80,   211,   213,   197,     0,     0,
       0,   250,     0,   249,     0,    29,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   339,   337,   364,     0,
       0,    53,    52,     0,     0,    59,    62,    57,    60,    83,
      86,    85,    67,    69,    66,    68,    92,     0,     0,   123,
     122,     7,   154,   153,   174,     0,     0,   179,   175,   195,
     194,    27,     0,    26,     0,   310,   309,   307,     0,   304,
       0,   209,   403,   207,     0,    22,    20,    21,   220,   219,
     227,     0,   246,   248,   224,   221,     0,   230,   229,     0,
     234,     0,   233,   238,     0,   237,     0,    25,     0,   206,
     210,   210,   100,    99,   414,   413,   421,   384,   385,     0,
     415,     0,     0,   411,   410,     0,   416,     0,   106,   103,
     105,   101,     0,     0,     0,     0,     0,     0,   215,   217,
       0,    80,     0,   201,   203,     0,   254,   252,     0,     0,
       0,   244,   391,     0,     0,     0,   368,   378,   382,   383,
     381,   380,   379,   377,   376,   375,   374,   373,   371,   408,
       0,   347,   346,   345,   344,   343,   342,   336,   341,   353,
     352,   351,   350,   349,   332,   331,   330,   325,   324,   328,
     327,   326,   329,   334,   333,     0,   419,     0,    50,   423,
       0,     0,   173,     0,     0,   206,   275,   263,     0,     0,
       0,   296,   303,     0,   404,     0,     0,     0,     0,     0,
     351,   232,   232,    18,   241,     0,   242,   240,   398,     0,
      80,    80,   387,   386,     0,   424,   417,     0,     0,    81,
       0,     0,     0,    72,    71,    76,     0,   127,     0,     0,
     125,     0,   137,     0,   158,     0,     0,   156,     0,     0,
     181,   182,    80,   216,   218,     0,     0,   214,     0,   199,
       0,   251,   243,   253,   392,   390,     0,   366,     0,   407,
       0,    30,     0,     0,    91,    87,    89,   172,   258,     0,
     285,     0,   295,   268,   264,   265,   294,   285,   306,   305,
     208,   402,   226,   225,   223,   222,    19,   397,     0,     0,
       0,     0,   388,     0,    55,     0,    74,     0,     0,     0,
      80,    80,   126,     0,   150,   148,   146,   147,     0,   144,
     141,     0,     0,   157,     0,   170,   171,     0,   167,     0,
       0,   185,   192,   193,     0,     0,   190,     0,   183,     0,
     196,     0,     0,     0,   198,   202,     0,   367,   372,   406,
     405,     0,    51,     0,     0,   261,   260,   277,     0,     0,
       0,     0,     0,   278,   276,   280,   279,     0,   257,     0,
     267,     0,   298,   299,   301,   300,     0,   297,   396,   395,
     400,     0,   425,    75,    79,    78,    64,     0,     0,   132,
     134,     0,   128,   130,     0,   121,    80,     0,   138,   163,
     165,   159,   161,   169,   152,   189,     0,   187,     0,     0,
     177,   212,   204,     0,   393,    31,     0,     0,     0,    95,
       0,     0,    96,    97,    98,    90,     0,     0,   281,     0,
       0,   288,     0,     0,     0,   273,   274,     0,   270,   272,
     266,     0,   389,    77,    80,     0,   149,    80,     0,   145,
       0,   143,    80,     0,    80,     0,   168,   186,   191,     0,
     200,     0,   108,     0,     0,   112,     0,     0,   116,     0,
       0,    94,   262,     0,   210,     0,   287,   289,   286,     0,
     256,   269,     0,   293,     0,   135,     0,   131,     0,   166,
       0,   162,   188,   111,    80,   110,   115,    80,   114,   119,
      80,   118,    88,   284,    80,     0,   290,     0,   271,     0,
       0,     0,     0,   283,   291,     0,     0,     0,     0,   109,
     113,   117,   282
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    59,    60,   567,    61,   479,    63,   119,
      64,    65,   213,    66,    67,    68,   218,    69,    70,   482,
     560,   483,   484,   561,   485,   372,    71,    72,    73,   603,
     604,   670,   671,    74,    75,    76,   672,   744,   673,   747,
     674,   750,    77,   220,    78,   374,   490,   697,   698,   694,
     695,   491,   572,   492,   648,   568,   569,    79,   221,    80,
     375,   497,   704,   705,   702,   703,   577,   578,    81,    82,
     222,    83,   499,   500,   501,   502,   585,   586,    84,    85,
      86,   593,    87,   508,    88,   322,   323,   224,   381,   382,
     225,   226,    89,    90,    91,    92,   172,    93,   176,    94,
     179,   180,    95,    96,    97,   232,   233,    98,   312,   446,
     447,   676,   450,   534,   535,   620,   687,   688,   530,   614,
     615,   724,   616,   617,   683,    99,   314,   451,   537,   627,
     100,   154,   318,   319,   101,   102,   103,   104,   105,   106,
     107,   596,   108,   183,   350,   109,   184,   110,   155,   324,
     111,   112,   113,   114,   115,   189,   133,   197
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -465
static const yytype_int16 yypact[] =
{
    -465,    58,   722,  -465,    64,  -465,  -465,  -465,  -465,  -465,
    -465,    72,   178,  3103,   158,   172,  3165,   192,  3227,    97,
    3289,  -465,  -465,  3351,   276,  3413,   387,   454,  2855,  -465,
    -465,   457,  3475,   485,   331,  3537,   467,   497,   531,   220,
    -465,  3599,  4980,    45,  -465,  -465,  -465,  5228,  4855,  5228,
    2917,  5228,  5228,  5228,  2979,  5228,  5228,  5228,    13,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,  -465,  2793,  -465,  -465,
    2793,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,   109,  2793,    59,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,    80,   181,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  4240,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,    93,  -465,  -465,   175,    79,
     120,   163,  -465,  4020,   286,  -465,   287,  -465,   324,   167,
    4079,   333,  -465,   -19,   336,  4291,   345,   348,  4342,   349,
    5703,    50,   351,  -465,  2793,   370,  4393,   374,  -465,   386,
    -465,   413,  -465,  4444,   270,    87,   436,   438,   439,   445,
    5703,   451,   453,   232,   173,  -465,  -465,   459,  4495,   498,
    -465,  -465,    84,   509,   516,   241,   517,   530,   468,    88,
    -465,   540,  -465,   264,   264,   543,  4546,  -465,  5703,  3041,
    -465,  5448,  5042,  -465,   490,  5280,    95,   137,  5958,   548,
    -465,    92,   268,   268,   518,   554,  -465,   141,   518,   518,
     518,  -465,  -465,  -465,   568,   574,   273,  -465,  -465,  -465,
    -465,  -465,  -465,   332,  -465,  -465,  -465,  -465,   582,   147,
     585,  -465,   199,    41,   208,  -465,  5104,  4918,   587,  5228,
    5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,
    5228,  3661,  5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,
     588,  5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,  5228,
    5228,  5228,  5228,  5228,  5228,  5228,  -465,  -465,  -465,  5228,
    5228,  -465,  -465,   590,  5228,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,  -465,   590,  3723,  -465,
    -465,  -465,  -465,  -465,  -465,   597,  5228,  -465,  -465,  -465,
    -465,  -465,    31,  -465,   279,  -465,  -465,  -465,   209,  -465,
     601,  -465,   536,  -465,   552,  -465,  -465,  -465,  -465,  -465,
    -465,   303,  -465,  -465,  -465,  -465,  3785,  -465,  -465,   610,
    -465,   611,  -465,  -465,    27,  -465,   612,  -465,   616,   614,
     109,   109,  -465,  -465,  -465,  -465,  5703,  -465,  -465,  5499,
    -465,  5166,  5228,  -465,  -465,   562,  -465,  5228,  -465,  -465,
    -465,  -465,  1594,   555,   442,   458,  1376,   278,  -465,  -465,
    1703,  -465,  2793,  -465,  -465,   115,  -465,  -465,   617,   619,
     212,  -465,  -465,   133,  5228,  5338,  -465,  5703,  5703,  5703,
    5703,  5703,  5703,  5703,  5703,  5703,  5703,  5703,  5754,  -465,
    3961,  5907,  5958,   427,   427,   427,   427,   427,   427,  -465,
     268,   268,   268,   268,   576,   576,  3873,   376,   376,   283,
     283,   283,   392,   518,   518,  4130,  5805,   550,  5703,  -465,
     624,  4189,  -465,  4597,   627,   614,  -465,   598,   629,   632,
     630,  -465,  -465,   539,  -465,   614,  5228,   637,   638,   639,
      11,  -465,   640,  -465,  -465,   642,  -465,  -465,  -465,   140,
    -465,  -465,  -465,  -465,  5396,  5703,  -465,  5550,   641,  -465,
     302,  3847,   644,  -465,  -465,  -465,   647,  -465,    48,   337,
    -465,   645,  -465,   648,  -465,   127,   646,  -465,    73,   650,
     626,  -465,  -465,  -465,  -465,   654,  1812,  -465,   604,  -465,
     298,  -465,  -465,  -465,  -465,  -465,  5601,  -465,  5228,  -465,
    3909,  -465,  5228,  5228,  -465,  -465,  -465,  -465,  -465,   207,
      91,   663,  -465,   609,   592,  -465,  -465,    98,  -465,  -465,
    -465,  5703,  -465,  -465,  -465,  -465,  -465,  -465,   667,  1921,
    2030,  5228,  -465,  5228,  -465,   668,  -465,   678,  4648,   679,
    -465,  -465,  -465,   312,  -465,  -465,  -465,   613,   245,  -465,
    -465,   682,   327,  -465,   334,  -465,  -465,   266,  -465,   683,
     684,  -465,  -465,  -465,   590,    14,  -465,   686,  -465,  1485,
    -465,   687,   651,   634,  -465,  -465,   635,  -465,   618,  -465,
    5856,   215,  5703,   831,  2793,  -465,  -465,  -465,   621,   694,
     693,   696,    20,  -465,  -465,  -465,  -465,   691,  -465,   292,
    -465,   632,  -465,  -465,  -465,  -465,   695,  -465,  -465,  -465,
    -465,  5652,  5703,  -465,  -465,  -465,  -465,  2139,   555,  -465,
    -465,    24,  -465,  -465,    55,  -465,  -465,  2793,  -465,  -465,
    -465,  -465,  -465,   310,  -465,  -465,   702,  -465,   314,   590,
    -465,  -465,  -465,   703,  -465,  -465,   446,   489,   506,  -465,
     698,   831,  -465,  -465,  -465,  -465,   652,  5228,  -465,   636,
     705,  -465,   706,   219,   712,  -465,  -465,   -30,  -465,  -465,
    -465,   713,  -465,  -465,  -465,  2793,  -465,  -465,  2793,  -465,
    2248,  -465,  -465,  2793,  -465,  2793,  -465,  -465,  -465,   715,
    -465,   716,  -465,  2793,   718,  -465,  2793,   721,  -465,  2793,
     733,  -465,  -465,  4699,   109,  5228,  -465,  -465,  -465,    19,
    -465,  -465,   292,  -465,   940,  -465,  1049,  -465,  1158,  -465,
    1267,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  4750,  -465,   732,  -465,  2357,
    2466,  2575,  2684,  -465,  -465,   736,   738,   739,   740,  -465,
    -465,  -465,  -465
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -465,  -465,  -465,  -465,  -465,  -332,  -465,    -2,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,   108,
    -465,  -465,  -465,  -465,  -465,  -200,  -465,  -465,  -465,  -465,
    -465,    89,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,  -465,
    -465,   380,  -465,  -465,  -465,  -465,   125,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,  -465,   118,  -465,  -465,
    -465,  -465,  -465,  -465,   281,  -465,  -465,   121,  -465,  -464,
    -465,  -465,  -465,  -465,  -465,  -226,   328,  -341,  -465,  -465,
    -465,  -465,  -465,  -465,  -465,  -465,   747,  -465,  -465,  -465,
    -465,   440,  -465,  -465,  -465,   -82,  -465,  -465,  -465,  -465,
    -465,  -465,   338,  -465,   166,  -465,  -465,    56,  -465,  -465,
     252,  -465,   253,   254,  -465,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,   339,  -465,  -326,   -10,  -465,   -12,    -3,
     761,  -465,  -465,  -465,   615,  -465,  -465,  -465,  -465,  -465,
    -465,  -465,  -465,  -465,    30,  -465,  -465,  -465
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -395
static const yytype_int16 yytable[] =
{
      62,   123,   120,   385,   130,   458,   135,   132,   138,   470,
     471,   140,   466,   146,   545,   234,   153,   657,   373,     8,
     160,   680,   376,   168,   380,   756,   681,   731,   463,   186,
     188,   463,   444,   464,  -259,   191,   195,   198,   140,   202,
     203,   204,   140,   208,   209,   210,   732,   231,   212,   563,
     389,   305,   463,   141,   564,   565,   566,   297,     3,   463,
     228,   564,   565,   566,  -259,   217,   613,   116,   219,   236,
     298,   237,   238,   623,   580,   117,   581,   582,   196,   583,
     201,   230,   282,  -249,   207,   227,   231,   338,   320,   445,
     658,   345,   190,   321,   607,   369,   363,   608,   136,   757,
     682,   622,   278,   659,   608,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   509,   229,   276,   277,
     278,   465,   211,   469,   465,   306,   280,   278,   574,   609,
    -169,   575,   278,   576,   514,   278,   609,   278,   365,   610,
     611,   547,   308,   278,   371,   465,   610,   611,   384,  -206,
     278,   390,   465,   321,   364,   283,  -249,   278,   223,   124,
     339,   125,   584,  -206,   346,   278,   285,   279,   280,   280,
     292,   280,   510,   126,  -169,   127,   332,   356,   281,   118,
     359,   506,   612,   278,     8,   278,   128,   231,   278,   612,
     515,   455,   278,   131,   284,   278,   366,   548,     8,   278,
     278,   278,   387,  -169,  -206,   278,   278,   278,   605,   280,
     286,   391,   452,   367,   293,   513,   455,   280,   665,   529,
     333,   181,   728,  -206,   140,   395,   182,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   406,   407,   408,   410,
     411,   412,   413,   414,   415,   416,   417,   418,   642,   420,
     421,   422,   423,   424,   425,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   606,   348,   393,   435,   436,   651,
     549,   550,   438,   437,   181,   388,   317,   142,  -394,   143,
     448,   503,  -263,   455,   388,   453,   441,   439,   388,   289,
     290,   280,   643,   689,   443,   729,     6,     7,   685,     9,
      10,   594,   589,   555,   457,   556,   331,     6,     7,   696,
       9,    10,   449,   652,   575,   639,   576,   341,   582,   686,
     583,   644,   349,   144,   460,   504,   236,   291,   237,   238,
     646,  -394,   164,   377,   165,   378,   296,   649,   570,   299,
    -140,   236,   653,   237,   238,   595,    44,    45,   301,   474,
     475,   302,   304,   278,   307,   477,   278,    44,    45,   640,
     637,   638,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   309,   647,   276,   277,   311,   166,   379,
     507,   650,   516,   754,  -140,   273,   274,   275,   147,   313,
     276,   277,   278,   148,   278,   278,   278,   278,   278,   278,
     278,   278,   278,   278,   278,   278,   689,   278,   278,   278,
     278,   278,   278,   278,   278,   278,   315,   278,   278,   278,
     278,   278,   278,   278,   278,   278,   278,   278,   278,   278,
     278,   278,   278,   278,   236,   278,   237,   238,   278,   325,
     278,   326,   327,   486,   541,   487,   700,   711,   328,   712,
     236,  -136,   237,   238,   329,   149,   330,   278,   156,   493,
     150,   494,   334,   157,   158,   488,   489,  -136,   169,   558,
     170,   278,   278,   171,   278,   270,   271,   272,   273,   274,
     275,   495,   489,   276,   277,   236,   162,   237,   238,  -139,
     714,   163,   715,   713,   734,   274,   275,   736,   173,   276,
     277,   337,   738,   174,   740,  -139,   140,   717,   600,   718,
     140,   602,   340,   278,   260,   261,   262,   263,   264,  -235,
     342,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   177,   343,   276,   277,   716,   178,   278,   631,
     538,   632,   344,   347,   759,   317,   352,   760,   598,   360,
     761,   368,   601,   719,   762,   278,     4,   370,     5,     6,
       7,     8,     9,    10,   -70,    12,    13,    14,    15,   147,
      16,   480,   481,    17,   656,   149,   236,    18,   237,   238,
      20,    21,    22,    23,    24,   383,    25,   214,   386,   215,
      28,    29,    30,   396,   419,    32,     8,   278,    35,   278,
     442,   669,   675,   216,   454,    40,    41,    42,    43,    44,
      45,    46,   455,    47,   456,    48,   461,   462,   178,   468,
     321,   476,   512,   511,   523,   276,   277,   524,   278,   278,
     528,   449,   532,   536,   236,    49,   237,   238,   533,    50,
     542,   543,   544,  -236,   554,   701,   546,    51,    52,   709,
     562,   573,    53,   559,   571,   579,   498,   590,    54,   587,
      55,   592,    56,    57,    58,   723,   618,   619,   621,   669,
     628,   633,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   634,   636,   276,   277,   645,   654,   655,   641,   660,
     661,   663,   664,   735,   280,   677,   737,   678,   662,   182,
     684,   739,   679,   741,   691,   707,   710,   720,   726,   722,
     725,   745,   727,   755,   748,   730,   733,   751,   742,   743,
     278,   746,    -2,     4,   749,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,   752,    16,   764,   769,
      17,   770,   771,   772,    18,    19,   693,    20,    21,    22,
      23,    24,   278,    25,    26,   496,    27,    28,    29,    30,
     721,    31,    32,    33,    34,    35,    36,    37,    38,   699,
      39,   706,    40,    41,    42,    43,    44,    45,    46,   708,
      47,   588,    48,   540,   175,   531,   467,   690,   758,   624,
     625,   626,   539,   161,     0,     0,     0,     0,     0,   351,
       0,     0,    49,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
     -93,    12,    13,    14,    15,     0,    16,     0,     0,    17,
     666,   667,   668,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   214,     0,   215,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   216,
       0,    40,    41,    42,    43,    44,    45,    46,     0,    47,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,  -133,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -133,  -133,    20,    21,    22,    23,    24,
       0,    25,   214,     0,   215,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,  -133,   216,     0,
      40,    41,    42,    43,    44,    45,    46,     0,    47,     0,
      48,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
       4,     0,     5,     6,     7,     8,     9,    10,  -129,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -129,  -129,    20,    21,    22,    23,    24,     0,
      25,   214,     0,   215,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,  -129,   216,     0,    40,
      41,    42,    43,    44,    45,    46,     0,    47,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,  -164,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,  -164,  -164,    20,    21,    22,    23,    24,     0,    25,
     214,     0,   215,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,  -164,   216,     0,    40,    41,
      42,    43,    44,    45,    46,     0,    47,     0,    48,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,  -160,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -160,  -160,    20,    21,    22,    23,    24,     0,    25,   214,
       0,   215,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,  -160,   216,     0,    40,    41,    42,
      43,    44,    45,    46,     0,    47,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,  -180,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,   498,    25,   214,     0,
     215,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   216,     0,    40,    41,    42,    43,
      44,    45,    46,     0,    47,     0,    48,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,  -184,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,  -184,    25,   214,     0,   215,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   216,     0,    40,    41,    42,    43,    44,
      45,    46,     0,    47,     0,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,   478,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   214,     0,   215,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   216,     0,    40,    41,    42,    43,    44,    45,
      46,     0,    47,     0,    48,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,   505,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   214,     0,   215,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   216,     0,    40,    41,    42,    43,    44,    45,    46,
       0,    47,     0,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,     4,     0,     5,     6,     7,     8,     9,
      10,   591,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   214,     0,   215,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     216,     0,    40,    41,    42,    43,    44,    45,    46,     0,
      47,     0,    48,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
     629,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   214,     0,   215,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   216,
       0,    40,    41,    42,    43,    44,    45,    46,     0,    47,
       0,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,   630,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   214,     0,   215,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   216,     0,
      40,    41,    42,    43,    44,    45,    46,     0,    47,     0,
      48,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
       4,     0,     5,     6,     7,     8,     9,    10,   -73,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   214,     0,   215,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   216,     0,    40,
      41,    42,    43,    44,    45,    46,     0,    47,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,  -142,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     214,     0,   215,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   216,     0,    40,    41,
      42,    43,    44,    45,    46,     0,    47,     0,    48,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,   765,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   214,
       0,   215,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   216,     0,    40,    41,    42,
      43,    44,    45,    46,     0,    47,     0,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,   766,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   214,     0,
     215,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   216,     0,    40,    41,    42,    43,
      44,    45,    46,     0,    47,     0,    48,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,   767,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   214,     0,   215,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   216,     0,    40,    41,    42,    43,    44,
      45,    46,     0,    47,     0,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,   768,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   214,     0,   215,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   216,     0,    40,    41,    42,    43,    44,    45,
      46,     0,    47,     0,    48,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,     0,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   214,     0,   215,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   216,     0,    40,    41,    42,    43,    44,    45,    46,
       0,    47,     0,    48,     0,     0,   151,     0,   152,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,    50,     0,     0,
       0,    21,    22,     0,     0,    51,    52,     0,     0,     0,
      53,     0,    30,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,   122,     0,    40,     0,    42,     0,    44,
      45,     0,     0,    47,     0,    48,     0,     0,   199,     0,
     200,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    51,    52,     0,
       0,     0,    53,     0,    30,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   122,     0,    40,     0,    42,
       0,    44,    45,     0,     0,    47,     0,    48,     0,     0,
     205,     0,   206,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,    51,
      52,     0,     0,     0,    53,     0,    30,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,   122,     0,    40,
       0,    42,     0,    44,    45,     0,     0,    47,     0,    48,
       0,     0,   354,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,    51,    52,     0,     0,     0,    53,     0,    30,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,   122,
       0,    40,     0,    42,     0,    44,    45,     0,     0,    47,
     355,    48,     0,     0,   121,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    51,    52,     0,     0,     0,    53,     0,
      30,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,   122,     0,    40,     0,    42,     0,    44,    45,     0,
       0,    47,     0,    48,     0,     0,   129,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,    51,    52,     0,     0,     0,
      53,     0,    30,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,   122,     0,    40,     0,    42,     0,    44,
      45,     0,     0,    47,     0,    48,     0,     0,   134,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    51,    52,     0,
       0,     0,    53,     0,    30,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   122,     0,    40,     0,    42,
       0,    44,    45,     0,     0,    47,     0,    48,     0,     0,
     137,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,    51,
      52,     0,     0,     0,    53,     0,    30,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,   122,     0,    40,
       0,    42,     0,    44,    45,     0,     0,    47,     0,    48,
       0,     0,   139,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,    51,    52,     0,     0,     0,    53,     0,    30,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,   122,
       0,    40,     0,    42,     0,    44,    45,     0,     0,    47,
       0,    48,     0,     0,   145,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    51,    52,     0,     0,     0,    53,     0,
      30,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,   122,     0,    40,     0,    42,     0,    44,    45,     0,
       0,    47,     0,    48,     0,     0,   159,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,    51,    52,     0,     0,     0,
      53,     0,    30,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,   122,     0,    40,     0,    42,     0,    44,
      45,     0,     0,    47,     0,    48,     0,     0,   167,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    51,    52,     0,
       0,     0,    53,     0,    30,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   122,     0,    40,     0,    42,
       0,    44,    45,     0,     0,    47,     0,    48,     0,     0,
     185,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,    51,
      52,     0,     0,     0,    53,     0,    30,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,   122,     0,    40,
       0,    42,     0,    44,    45,     0,     0,    47,     0,    48,
       0,     0,   409,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,    51,    52,     0,     0,     0,    53,     0,    30,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,   122,
       0,    40,     0,    42,     0,    44,    45,     0,     0,    47,
       0,    48,     0,     0,   440,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    51,    52,     0,     0,     0,    53,     0,
      30,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,   122,     0,    40,     0,    42,     0,    44,    45,     0,
       0,    47,     0,    48,     0,     0,   459,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,    51,    52,     0,     0,     0,
      53,     0,    30,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,   122,     0,    40,     0,    42,     0,    44,
      45,     0,     0,    47,     0,    48,     0,     0,   557,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    51,    52,     0,
       0,     0,    53,     0,    30,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   122,     0,    40,     0,    42,
       0,    44,    45,     0,     0,    47,     0,    48,     0,     0,
     599,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
       0,   236,     0,   237,   238,    21,    22,     0,     0,    51,
      52,     0,     0,     0,    53,     0,    30,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,   122,     0,    40,
       0,    42,   519,    44,    45,     0,     0,    47,     0,    48,
     268,   269,   270,   271,   272,   273,   274,   275,     0,     0,
     276,   277,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,   520,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,   236,
       0,   237,   238,   287,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,     0,     0,   251,   252,
     253,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,     0,   288,   276,   277,
       0,     0,     0,     0,     0,     0,     0,     0,   236,     0,
     237,   238,   294,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,     0,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,   295,   276,   277,     0,
       0,     0,     0,   521,     0,     0,     0,   236,     0,   237,
     238,     0,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,     0,     0,   251,   252,   253,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,   276,   277,   236,     0,
     237,   238,   525,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,   522,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,   526,   276,   277,     0,
       0,     0,     0,   235,     0,     0,     0,   236,     0,   237,
     238,     0,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,     0,     0,   251,   252,   253,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   300,     0,   276,   277,   236,     0,
     237,   238,     0,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,     0,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   303,     0,   276,   277,   236,
       0,   237,   238,     0,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,     0,     0,   251,   252,
     253,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   310,     0,   276,   277,
     236,     0,   237,   238,     0,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,     0,     0,   251,
     252,   253,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   316,     0,   276,
     277,   236,     0,   237,   238,     0,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,     0,     0,
     251,   252,   253,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,     0,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   335,     0,
     276,   277,   236,     0,   237,   238,     0,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,     0,
       0,   251,   252,   253,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   353,
       0,   276,   277,   236,     0,   237,   238,     0,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
       0,     0,   251,   252,   253,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   336,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     527,     0,   276,   277,   236,     0,   237,   238,     0,   239,
     240,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,     0,     0,   251,   252,   253,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   635,     0,   276,   277,   236,     0,   237,   238,     0,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,     0,     0,   251,   252,   253,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   753,     0,   276,   277,   236,     0,   237,   238,
       0,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,     0,     0,   251,   252,   253,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   763,     0,   276,   277,   236,     0,   237,
     238,     0,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,     0,     0,   251,   252,   253,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,   276,   277,   236,     0,
     237,   238,     0,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,     0,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   276,   277,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   192,   122,     0,    40,     0,    42,     0,    44,
      45,     0,     0,    47,   193,    48,     0,   194,     0,     0,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,    51,    52,     0,
       0,     0,    53,     0,     0,    30,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   192,   122,     0,    40,     0,
      42,     0,    44,    45,     0,     0,    47,     0,    48,     0,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
      51,    52,     0,     0,     0,    53,     0,    30,     0,   394,
       0,     0,     0,    55,     0,    56,    57,    58,   122,     0,
      40,     0,    42,     0,    44,    45,     0,     0,    47,   187,
      48,     0,     0,     0,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,    51,    52,     0,     0,     0,    53,     0,    30,
       0,     0,     0,     0,     0,    55,     0,    56,    57,    58,
     122,     0,    40,     0,    42,     0,    44,    45,     0,     0,
      47,   358,    48,     0,     0,     0,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,    51,    52,     0,     0,     0,    53,
       0,    30,     0,     0,     0,     0,     0,    55,     0,    56,
      57,    58,   122,     0,    40,     0,    42,     0,    44,    45,
       0,   392,    47,     0,    48,     0,     0,     0,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    51,    52,     0,     0,
       0,    53,     0,    30,     0,     0,     0,     0,     0,    55,
       0,    56,    57,    58,   122,     0,    40,     0,    42,     0,
      44,    45,     0,     0,    47,   473,    48,     0,     0,     0,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,    51,    52,
       0,     0,     0,    53,     0,    30,     0,     0,     0,     0,
       0,    55,     0,    56,    57,    58,   122,     0,    40,     0,
      42,     0,    44,    45,     0,     0,    47,     0,    48,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,   361,     0,     0,
       0,     0,     0,    55,     0,    56,    57,    58,   236,     0,
     237,   238,   362,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,     0,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   361,     0,   276,   277,     0,
       0,     0,     0,     0,     0,     0,   236,   517,   237,   238,
       0,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,     0,     0,   251,   252,   253,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   551,     0,   276,   277,     0,     0,     0,
       0,     0,     0,     0,   236,   552,   237,   238,     0,   239,
     240,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,     0,     0,   251,   252,   253,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,   357,   236,     0,   237,   238,
       0,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,     0,     0,   251,   252,   253,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277,   236,   472,   237,
     238,     0,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,     0,     0,   251,   252,   253,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,   276,   277,   236,     0,
     237,   238,   553,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,     0,     0,   251,   252,   253,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   276,   277,   236,
     597,   237,   238,     0,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,     0,     0,   251,   252,
     253,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,     0,     0,   276,   277,
     236,   692,   237,   238,     0,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,     0,     0,   251,
     252,   253,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,   276,
     277,   236,     0,   237,   238,     0,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,     0,     0,
     251,   252,   253,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,     0,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,     0,     0,
     276,   277,   236,     0,   237,   238,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   250,     0,
     518,   251,   252,   253,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,     0,
       0,   276,   277,   236,     0,   237,   238,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   251,   252,   253,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
       0,     0,   276,   277,   236,     0,   237,   238,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   252,   253,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,   236,     0,   237,   238,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   253,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,     0,     0,   276,   277,   236,     0,   237,   238,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   229,    16,   331,    18,    17,    20,   350,
     351,    23,   344,    25,     3,    97,    28,     3,   218,     6,
      32,     1,   222,    35,   224,     6,     6,    57,     4,    41,
      42,     4,     1,     6,     3,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    76,     6,    58,     1,
       9,     1,     4,    23,     6,     7,     8,    76,     0,     4,
       1,     6,     7,     8,    33,    67,   530,     3,    70,    58,
      89,    60,    61,   537,     1,     3,     3,     4,    48,     6,
      50,     1,     3,     3,    54,    87,     6,     3,     1,    58,
      76,     3,    47,     6,     3,     3,     1,     6,     1,    80,
      80,     3,   105,    89,     6,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,     1,    58,   107,   108,
     123,    97,   109,   349,    97,    75,    76,   130,     1,    38,
       3,     4,   135,     6,     1,   138,    38,   140,     1,    48,
      49,     1,   144,   146,     3,    97,    48,    49,     1,    62,
     153,   233,    97,     6,    59,    76,    76,   160,    49,     1,
      76,     3,    89,    76,    76,   168,     3,    74,    76,    76,
       3,    76,    57,     1,    47,     3,     3,   189,     3,     1,
     192,   381,    91,   186,     6,   188,    14,     6,   191,    91,
      57,    76,   195,     1,    74,   198,    59,    57,     6,   202,
     203,   204,     3,    76,    57,   208,   209,   210,     1,    76,
      47,     3,     3,    76,    47,     3,    76,    76,     3,   445,
      47,     1,     3,    76,   236,   237,     6,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,     3,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,    57,     1,   236,   279,   280,     3,
     470,   471,   284,   283,     1,    76,     6,     1,    58,     3,
       1,     3,     3,    76,    76,    76,   298,   297,    76,     3,
       3,    76,    47,   619,   306,    76,     4,     5,     6,     7,
       8,     3,   502,     1,     1,     3,    74,     4,     5,   641,
       7,     8,    33,    47,     4,     3,     6,    76,     4,    27,
       6,    76,    58,    47,   336,    47,    58,     3,    60,    61,
       3,    58,     1,     1,     3,     3,     3,     3,     1,     3,
       3,    58,    76,    60,    61,    47,    54,    55,     3,   361,
     362,     3,     3,   356,     3,   367,   359,    54,    55,    47,
     560,   561,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,     3,    47,   107,   108,     3,    47,    47,
     382,    47,   394,   724,    47,   102,   103,   104,     1,     3,
     107,   108,   395,     6,   397,   398,   399,   400,   401,   402,
     403,   404,   405,   406,   407,   408,   732,   410,   411,   412,
     413,   414,   415,   416,   417,   418,     3,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,    58,   438,    60,    61,   441,     3,
     443,     3,     3,     1,   456,     3,   646,     1,     3,     3,
      58,     9,    60,    61,     3,     1,     3,   460,     1,     1,
       6,     3,     3,     6,     7,    23,    24,     9,     1,   481,
       3,   474,   475,     6,   477,    99,   100,   101,   102,   103,
     104,    23,    24,   107,   108,    58,     1,    60,    61,    47,
       1,     6,     3,    47,   694,   103,   104,   697,     1,   107,
     108,     3,   702,     6,   704,    47,   518,     1,   520,     3,
     522,   523,     3,   516,    87,    88,    89,    90,    91,     3,
       3,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,     1,     3,   107,   108,    47,     6,   541,   551,
       1,   553,    74,     3,   744,     6,     3,   747,   518,    59,
     750,     3,   522,    47,   754,   558,     1,     3,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,     1,
      15,    16,    17,    18,   584,     1,    58,    22,    60,    61,
      25,    26,    27,    28,    29,     3,    31,    32,     3,    34,
      35,    36,    37,     6,     6,    40,     6,   600,    43,   602,
       3,   603,   604,    48,     3,    50,    51,    52,    53,    54,
      55,    56,    76,    58,    62,    60,     6,     6,     6,     3,
       6,    59,     3,     6,    74,   107,   108,     3,   631,   632,
       3,    33,     3,     3,    58,    80,    60,    61,     6,    84,
       3,     3,     3,     3,     3,   647,     4,    92,    93,   659,
       3,     3,    97,     9,     9,     9,    30,     3,   103,     9,
     105,    57,   107,   108,   109,   677,     3,    58,    76,   671,
       3,     3,    96,    97,    98,    99,   100,   101,   102,   103,
     104,     3,     3,   107,   108,     3,     3,     3,    75,     3,
       3,    57,    57,   695,    76,    74,   698,     3,    47,     6,
       9,   703,     6,   705,     9,     3,     3,     9,     3,    57,
      74,   713,     6,   725,   716,     3,     3,   719,     3,     3,
     723,     3,     0,     1,     3,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,     3,    15,     6,     3,
      18,     3,     3,     3,    22,    23,   638,    25,    26,    27,
      28,    29,   755,    31,    32,   375,    34,    35,    36,    37,
     671,    39,    40,    41,    42,    43,    44,    45,    46,   644,
      48,   653,    50,    51,    52,    53,    54,    55,    56,   658,
      58,   500,    60,   455,    37,   447,   346,   621,   732,   537,
     537,   537,   453,    32,    -1,    -1,    -1,    -1,    -1,   184,
      -1,    -1,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,    -1,   107,
     108,   109,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      19,    20,    21,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,
      -1,    -1,    -1,    -1,   103,    -1,   105,    -1,   107,   108,
     109,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,    -1,   107,   108,   109,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,    -1,
      -1,    -1,   103,    -1,   105,    -1,   107,   108,   109,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    23,    24,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    47,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,
      -1,   103,    -1,   105,    -1,   107,   108,   109,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,    -1,   107,   108,   109,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    30,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,
      -1,   105,    -1,   107,   108,   109,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,    -1,   107,   108,   109,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,   105,
      -1,   107,   108,   109,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    -1,    -1,    -1,    -1,    -1,   103,    -1,   105,    -1,
     107,   108,   109,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,    -1,   107,
     108,   109,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,
      -1,    -1,    -1,    -1,   103,    -1,   105,    -1,   107,   108,
     109,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,    -1,   107,   108,   109,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,    -1,
      -1,    -1,   103,    -1,   105,    -1,   107,   108,   109,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,
      -1,   103,    -1,   105,    -1,   107,   108,   109,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,    -1,   107,   108,   109,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,
      -1,   105,    -1,   107,   108,   109,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,    -1,   107,   108,   109,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,   105,
      -1,   107,   108,   109,     1,    -1,     3,     4,     5,     6,
       7,     8,    -1,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      -1,    58,    -1,    60,    -1,    -1,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    26,    27,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    -1,    37,    -1,    -1,    -1,   103,    -1,   105,    -1,
     107,   108,   109,    48,    -1,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    58,    -1,    60,    -1,    -1,     1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,    48,    -1,    50,    -1,    52,
      -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,   108,   109,    48,    -1,    50,
      -1,    52,    -1,    54,    55,    -1,    -1,    58,    -1,    60,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    37,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,   108,   109,    48,
      -1,    50,    -1,    52,    -1,    54,    55,    -1,    -1,    58,
      59,    60,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,   108,
     109,    48,    -1,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    58,    -1,    60,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    -1,    37,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,   108,   109,    48,    -1,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    58,    -1,    60,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,    48,    -1,    50,    -1,    52,
      -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,   108,   109,    48,    -1,    50,
      -1,    52,    -1,    54,    55,    -1,    -1,    58,    -1,    60,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    37,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,   108,   109,    48,
      -1,    50,    -1,    52,    -1,    54,    55,    -1,    -1,    58,
      -1,    60,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,   108,
     109,    48,    -1,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    58,    -1,    60,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    -1,    37,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,   108,   109,    48,    -1,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    58,    -1,    60,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,    48,    -1,    50,    -1,    52,
      -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,   108,   109,    48,    -1,    50,
      -1,    52,    -1,    54,    55,    -1,    -1,    58,    -1,    60,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    37,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,   108,   109,    48,
      -1,    50,    -1,    52,    -1,    54,    55,    -1,    -1,    58,
      -1,    60,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,   108,
     109,    48,    -1,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    58,    -1,    60,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    -1,    37,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,   108,   109,    48,    -1,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    58,    -1,    60,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,    48,    -1,    50,    -1,    52,
      -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    58,    -1,    60,    61,    26,    27,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,   108,   109,    48,    -1,    50,
      -1,    52,     1,    54,    55,    -1,    -1,    58,    -1,    60,
      97,    98,    99,   100,   101,   102,   103,   104,    -1,    -1,
     107,   108,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    47,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,   108,   109,    58,
      -1,    60,    61,     3,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    -1,    -1,    77,    78,
      79,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    47,   107,   108,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    58,    -1,
      60,    61,     3,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    47,   107,   108,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    -1,    58,    -1,    60,
      61,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    -1,    -1,    77,    78,    79,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,   107,   108,    58,    -1,
      60,    61,     3,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    76,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    47,   107,   108,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    -1,    58,    -1,    60,
      61,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    -1,    -1,    77,    78,    79,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,     3,    -1,   107,   108,    58,    -1,
      60,    61,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,     3,    -1,   107,   108,    58,
      -1,    60,    61,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    -1,    -1,    77,    78,
      79,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,     3,    -1,   107,   108,
      58,    -1,    60,    61,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    -1,    -1,    77,
      78,    79,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,     3,    -1,   107,
     108,    58,    -1,    60,    61,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    -1,    -1,
      77,    78,    79,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,     3,    -1,
     107,   108,    58,    -1,    60,    61,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    -1,
      -1,    77,    78,    79,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,     3,
      -1,   107,   108,    58,    -1,    60,    61,    -1,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      -1,    -1,    77,    78,    79,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
       3,    -1,   107,   108,    58,    -1,    60,    61,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    -1,    -1,    77,    78,    79,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,     3,    -1,   107,   108,    58,    -1,    60,    61,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    -1,    -1,    77,    78,    79,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,     3,    -1,   107,   108,    58,    -1,    60,    61,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    -1,    -1,    77,    78,    79,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,     3,    -1,   107,   108,    58,    -1,    60,
      61,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    -1,    -1,    77,    78,    79,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,   107,   108,    58,    -1,
      60,    61,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,   107,   108,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    47,    48,    -1,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    58,    59,    60,    -1,    62,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    37,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,    47,    48,    -1,    50,    -1,
      52,    -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    -1,    37,    -1,   101,
      -1,    -1,    -1,   105,    -1,   107,   108,   109,    48,    -1,
      50,    -1,    52,    -1,    54,    55,    -1,    -1,    58,    59,
      60,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    37,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,   108,   109,
      48,    -1,    50,    -1,    52,    -1,    54,    55,    -1,    -1,
      58,    59,    60,    -1,    -1,    -1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    37,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
     108,   109,    48,    -1,    50,    -1,    52,    -1,    54,    55,
      -1,    57,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    -1,    37,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,   108,   109,    48,    -1,    50,    -1,    52,    -1,
      54,    55,    -1,    -1,    58,    59,    60,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    -1,    37,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,   108,   109,    48,    -1,    50,    -1,
      52,    -1,    54,    55,    -1,    -1,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    -1,    47,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,   108,   109,    58,    -1,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    47,    -1,   107,   108,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    58,    59,    60,    61,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    -1,    -1,    77,    78,    79,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    47,    -1,   107,   108,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    58,    59,    60,    61,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    -1,    -1,    77,    78,    79,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    -1,   107,   108,    57,    58,    -1,    60,    61,
      -1,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    -1,    -1,    77,    78,    79,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    -1,    -1,   107,   108,    58,    59,    60,
      61,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    -1,    -1,    77,    78,    79,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,   107,   108,    58,    -1,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    77,    78,    79,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,   107,   108,    58,
      59,    60,    61,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    -1,    -1,    77,    78,
      79,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    -1,   107,   108,
      58,    59,    60,    61,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    -1,    -1,    77,
      78,    79,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    -1,    -1,   107,
     108,    58,    -1,    60,    61,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    -1,    -1,
      77,    78,    79,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,    -1,    -1,
     107,   108,    58,    -1,    60,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    -1,
      76,    77,    78,    79,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    -1,
      -1,   107,   108,    58,    -1,    60,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    77,    78,    79,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      -1,    -1,   107,   108,    58,    -1,    60,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    78,    79,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    -1,   107,   108,    58,    -1,    60,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    -1,    -1,   107,   108,    58,    -1,    60,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    -1,    -1,   107,   108
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   111,   112,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    58,    60,    80,
      84,    92,    93,    97,   103,   105,   107,   108,   109,   113,
     114,   116,   117,   118,   120,   121,   123,   124,   125,   127,
     128,   136,   137,   138,   143,   144,   145,   152,   154,   167,
     169,   178,   179,   181,   188,   189,   190,   192,   194,   202,
     203,   204,   205,   207,   209,   212,   213,   214,   217,   235,
     240,   244,   245,   246,   247,   248,   249,   250,   252,   255,
     257,   260,   261,   262,   263,   264,     3,     3,     1,   119,
     246,     1,    48,   248,     1,     3,     1,     3,    14,     1,
     248,     1,   246,   266,     1,   248,     1,     1,   248,     1,
     248,   264,     1,     3,    47,     1,   248,     1,     6,     1,
       6,     1,     3,   248,   241,   258,     1,     6,     7,     1,
     248,   250,     1,     6,     1,     3,    47,     1,   248,     1,
       3,     6,   206,     1,     6,   206,   208,     1,     6,   210,
     211,     1,     6,   253,   256,     1,   248,    59,   248,   265,
      47,   248,    47,    59,    62,   248,   264,   267,   248,     1,
       3,   264,   248,   248,   248,     1,     3,   264,   248,   248,
     248,   109,   246,   122,    32,    34,    48,   117,   126,   117,
     153,   168,   180,    49,   197,   200,   201,   117,     1,    58,
       1,     6,   215,   216,   215,     3,    58,    60,    61,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    77,    78,    79,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   107,   108,   249,    74,
      76,     3,     3,    76,    74,     3,    47,     3,    47,     3,
       3,     3,     3,    47,     3,    47,     3,    76,    89,     3,
       3,     3,     3,     3,     3,     1,    75,     3,   117,     3,
       3,     3,   218,     3,   236,     3,     3,     6,   242,   243,
       1,     6,   195,   196,   259,     3,     3,     3,     3,     3,
       3,    74,     3,    47,     3,     3,    89,     3,     3,    76,
       3,    76,     3,     3,    74,     3,    76,     3,     1,    58,
     254,   254,     3,     3,     1,    59,   248,    57,    59,   248,
      59,    47,    62,     1,    59,     1,    59,    76,     3,     3,
       3,     3,   135,   135,   155,   170,   135,     1,     3,    47,
     135,   198,   199,     3,     1,   195,     3,     3,    76,     9,
     215,     3,    57,   264,   101,   248,     6,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   248,   248,     1,
     248,   248,   248,   248,   248,   248,   248,   248,   248,     6,
     248,   248,   248,   248,   248,   248,   248,   248,   248,   248,
     248,   248,   248,   248,   248,   248,   248,   246,   248,   246,
       1,   248,     3,   248,     1,    58,   219,   220,     1,    33,
     222,   237,     3,    76,     3,    76,    62,     1,   245,     1,
     248,     6,     6,     4,     6,    97,   115,   211,     3,   195,
     197,   197,    59,    59,   248,   248,    59,   248,     9,   117,
      16,    17,   129,   131,   132,   134,     1,     3,    23,    24,
     156,   161,   163,     1,     3,    23,   161,   171,    30,   182,
     183,   184,   185,     3,    47,     9,   135,   117,   193,     1,
      57,     6,     3,     3,     1,    57,   248,    59,    76,     1,
      47,     3,    76,    74,     3,     3,    47,     3,     3,   195,
     228,   222,     3,     6,   223,   224,     3,   238,     1,   243,
     196,   248,     3,     3,     3,     3,     4,     1,    57,   135,
     135,    47,    59,    62,     3,     1,     3,     1,   248,     9,
     130,   133,     3,     1,     6,     7,     8,   115,   165,   166,
       1,     9,   162,     3,     1,     4,     6,   176,   177,     9,
       1,     3,     4,     6,    89,   186,   187,     9,   184,   135,
       3,     9,    57,   191,     3,    47,   251,    59,   264,     1,
     248,   264,   248,   139,   140,     1,    57,     3,     6,    38,
      48,    49,    91,   189,   229,   230,   232,   233,     3,    58,
     225,    76,     3,   189,   230,   232,   233,   239,     3,     9,
       9,   248,   248,     3,     3,     3,     3,   135,   135,     3,
      47,    75,     3,    47,    76,     3,     3,    47,   164,     3,
      47,     3,    47,    76,     3,     3,   246,     3,    76,    89,
       3,     3,    47,    57,    57,     3,    19,    20,    21,   117,
     141,   142,   146,   148,   150,   117,   221,    74,     3,     6,
       1,     6,    80,   234,     9,     6,    27,   226,   227,   245,
     224,     9,    59,   129,   159,   160,   115,   157,   158,   166,
     135,   117,   174,   175,   172,   173,   177,     3,   187,   246,
       3,     1,     3,    47,     1,     3,    47,     1,     3,    47,
       9,   141,    57,   248,   231,    74,     3,     6,     3,    76,
       3,    57,    76,     3,   135,   117,   135,   117,   135,   117,
     135,   117,     3,     3,   147,   117,     3,   149,   117,     3,
     151,   117,     3,     3,   197,   248,     6,    80,   227,   135,
     135,   135,   135,     3,     6,     9,     9,     9,     9,     3,
       3,     3,     3
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
        case 6:
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 19:
#line 248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); ;}
    break;

  case 20:
#line 253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 21:
#line 259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 22:
#line 265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 23:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 24:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 26:
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 27:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 28:
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 29:
#line 282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 30:
#line 284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
      COMPILER->defineVal( first );
      (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
         new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
   ;}
    break;

  case 31:
#line 290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
      {
         COMPILER->raiseError(Falcon::e_unpack_size );
      }
      Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

      COMPILER->defineVal( first );
      (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
      (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
         new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
   ;}
    break;

  case 50:
#line 326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   ;}
    break;

  case 51:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   ;}
    break;

  case 52:
#line 341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 53:
#line 343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 54:
#line 347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 55:
#line 354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 56:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 57:
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 58:
#line 370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 59:
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 60:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 61:
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 62:
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 63:
#line 381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 64:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 65:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 66:
#line 406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 67:
#line 407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 68:
#line 411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 69:
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 72:
#line 419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 75:
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 76:
#line 434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 78:
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 79:
#line 447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 81:
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 82:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 83:
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 84:
#line 476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 85:
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 86:
#line 495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 87:
#line 504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 88:
#line 520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 89:
#line 528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
          Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );

         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 90:
#line 544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 91:
#line 554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 92:
#line 559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 95:
#line 571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 99:
#line 585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      ;}
    break;

  case 100:
#line 598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 101:
#line 606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 102:
#line 610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 103:
#line 616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 104:
#line 622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 105:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 106:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 107:
#line 643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 108:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      ;}
    break;

  case 109:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 110:
#line 666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 111:
#line 675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 112:
#line 679 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      ;}
    break;

  case 113:
#line 691 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 114:
#line 692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 115:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 116:
#line 705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      ;}
    break;

  case 117:
#line 719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 118:
#line 721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 119:
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); ;}
    break;

  case 120:
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 121:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 122:
#line 751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 123:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 126:
#line 762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 128:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 130:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 131:
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 132:
#line 790 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 134:
#line 802 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 135:
#line 812 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 137:
#line 821 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      ;}
    break;

  case 141:
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 143:
#line 839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 146:
#line 851 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 147:
#line 860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 148:
#line 872 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 149:
#line 883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 150:
#line 894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 151:
#line 914 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 152:
#line 922 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 153:
#line 931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 154:
#line 933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 157:
#line 942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 159:
#line 948 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 161:
#line 958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 162:
#line 967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 163:
#line 971 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 983 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 166:
#line 993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 170:
#line 1007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 171:
#line 1019 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 172:
#line 1040 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 173:
#line 1044 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 174:
#line 1048 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 175:
#line 1056 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 176:
#line 1063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 177:
#line 1073 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 179:
#line 1082 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 185:
#line 1102 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 186:
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 187:
#line 1140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 188:
#line 1150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 189:
#line 1161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 192:
#line 1174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 193:
#line 1186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 194:
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 195:
#line 1209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 196:
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 197:
#line 1227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 199:
#line 1236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 200:
#line 1237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 201:
#line 1240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 203:
#line 1245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 204:
#line 1246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 205:
#line 1253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
            }
            else {
                cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 209:
#line 1314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      ;}
    break;

  case 211:
#line 1331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 212:
#line 1337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 213:
#line 1342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 214:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 216:
#line 1357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 218:
#line 1362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 219:
#line 1372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 220:
#line 1375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 221:
#line 1384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 222:
#line 1391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         if ( COMPILER->getFunction() == 0 )
         {
            COMPILER->raiseError(Falcon::e_pass_outside );
            /*delete $2;
            delete $4;*/
            (yyval.fal_stat) = 0;
         }
         else {
            COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         }
      ;}
    break;

  case 223:
#line 1406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 224:
#line 1412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 225:
#line 1424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 226:
#line 1434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 227:
#line 1439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 228:
#line 1451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 229:
#line 1460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 230:
#line 1467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 231:
#line 1475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 232:
#line 1480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 233:
#line 1488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 234:
#line 1492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 235:
#line 1500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 236:
#line 1505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 237:
#line 1517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 238:
#line 1522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 241:
#line 1535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 242:
#line 1539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 243:
#line 1553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 244:
#line 1560 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 246:
#line 1568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 248:
#line 1572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 250:
#line 1578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 251:
#line 1582 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 254:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 255:
#line 1602 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      ;}
    break;

  case 256:
#line 1636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      ;}
    break;

  case 258:
#line 1664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 261:
#line 1672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 262:
#line 1673 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 267:
#line 1690 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( (yyvsp[(2) - (2)].fal_adecl) != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = (yyvsp[(2) - (2)].fal_adecl)->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete (yyvsp[(2) - (2)].fal_adecl);
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      ;}
    break;

  case 268:
#line 1723 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 269:
#line 1728 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 270:
#line 1734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 271:
#line 1735 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 273:
#line 1741 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 274:
#line 1749 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 278:
#line 1759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 279:
#line 1762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 281:
#line 1784 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      ;}
    break;

  case 282:
#line 1808 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 283:
#line 1819 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   ;}
    break;

  case 284:
#line 1841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   ;}
    break;

  case 287:
#line 1871 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 288:
#line 1878 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 289:
#line 1886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 290:
#line 1892 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 291:
#line 1898 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 292:
#line 1911 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 293:
#line 1951 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 295:
#line 1976 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 299:
#line 1988 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 300:
#line 1991 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 302:
#line 2019 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 303:
#line 2024 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      ;}
    break;

  case 306:
#line 2039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 307:
#line 2046 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 308:
#line 2061 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 309:
#line 2062 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 310:
#line 2063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 311:
#line 2073 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 312:
#line 2074 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); ;}
    break;

  case 313:
#line 2075 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); ;}
    break;

  case 314:
#line 2076 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 315:
#line 2077 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 316:
#line 2078 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 317:
#line 2083 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     ;}
    break;

  case 319:
#line 2101 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 320:
#line 2102 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 323:
#line 2115 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 324:
#line 2116 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 325:
#line 2117 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 326:
#line 2118 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 327:
#line 2119 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 328:
#line 2120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 329:
#line 2121 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 330:
#line 2122 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 331:
#line 2123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 332:
#line 2124 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 333:
#line 2125 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 334:
#line 2126 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 335:
#line 2127 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 336:
#line 2128 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 337:
#line 2129 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 338:
#line 2130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 339:
#line 2131 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 340:
#line 2132 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 341:
#line 2133 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 342:
#line 2134 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 343:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 344:
#line 2136 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 345:
#line 2137 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 346:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 347:
#line 2139 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 348:
#line 2140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 349:
#line 2141 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2142 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 351:
#line 2143 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2144 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2145 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 354:
#line 2146 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 355:
#line 2147 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); ;}
    break;

  case 356:
#line 2148 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 357:
#line 2149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 364:
#line 2157 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 365:
#line 2162 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   ;}
    break;

  case 366:
#line 2166 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   ;}
    break;

  case 367:
#line 2171 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 368:
#line 2177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 371:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 372:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   ;}
    break;

  case 373:
#line 2201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 382:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 383:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 384:
#line 2212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);;}
    break;

  case 385:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 386:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 387:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 388:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 389:
#line 2229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      ;}
    break;

  case 390:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 391:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 392:
#line 2246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 393:
#line 2247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 394:
#line 2256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 395:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         ;}
    break;

  case 397:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 398:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 399:
#line 2309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 400:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         ;}
    break;

  case 401:
#line 2354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 402:
#line 2386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         ;}
    break;

  case 404:
#line 2398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 405:
#line 2407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 406:
#line 2412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 407:
#line 2419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 408:
#line 2426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 409:
#line 2435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); ;}
    break;

  case 410:
#line 2437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      ;}
    break;

  case 411:
#line 2441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      ;}
    break;

  case 412:
#line 2448 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 413:
#line 2450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 414:
#line 2454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 415:
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 416:
#line 2463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 417:
#line 2465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 418:
#line 2472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 419:
#line 2473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 420:
#line 2477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 421:
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 422:
#line 2482 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 423:
#line 2488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 424:
#line 2495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 425:
#line 2496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6152 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


