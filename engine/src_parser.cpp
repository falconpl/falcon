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




/* Copy the first part of user declarations.  */
#line 23 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


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
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 384 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   5984

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  171
/* YYNRULES -- Number of rules.  */
#define YYNRULES  437
/* YYNRULES -- Number of states.  */
#define YYNSTATES  830

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   361

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
     105,   106
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    38,    42,    46,
      48,    50,    54,    58,    62,    65,    69,    75,    78,    80,
      82,    84,    86,    88,    90,    92,    94,    96,    98,   100,
     102,   104,   106,   108,   110,   112,   114,   116,   118,   120,
     124,   128,   133,   139,   144,   149,   156,   163,   165,   167,
     169,   171,   173,   175,   177,   179,   181,   183,   185,   190,
     195,   200,   205,   210,   215,   220,   225,   230,   235,   240,
     241,   247,   250,   254,   256,   260,   264,   267,   271,   272,
     279,   282,   286,   290,   294,   298,   299,   301,   302,   306,
     309,   313,   314,   319,   323,   327,   328,   331,   334,   338,
     341,   345,   349,   350,   356,   359,   367,   377,   381,   389,
     399,   403,   404,   414,   421,   427,   428,   431,   433,   435,
     437,   439,   443,   447,   451,   454,   458,   461,   465,   467,
     468,   475,   479,   483,   484,   491,   495,   499,   500,   507,
     511,   515,   516,   523,   527,   531,   532,   535,   539,   541,
     542,   548,   549,   555,   556,   562,   563,   569,   570,   571,
     575,   576,   578,   581,   584,   587,   589,   593,   595,   597,
     599,   603,   605,   606,   613,   617,   621,   622,   625,   629,
     631,   632,   638,   639,   645,   646,   652,   653,   659,   661,
     665,   666,   668,   670,   676,   681,   685,   689,   690,   697,
     700,   704,   705,   707,   709,   712,   715,   718,   723,   727,
     733,   737,   739,   743,   745,   747,   751,   755,   761,   764,
     772,   773,   783,   787,   795,   796,   805,   808,   809,   811,
     816,   818,   819,   820,   826,   827,   831,   834,   838,   841,
     845,   849,   853,   857,   863,   869,   873,   879,   885,   889,
     892,   896,   900,   902,   906,   911,   915,   918,   922,   925,
     929,   930,   932,   936,   939,   943,   946,   947,   956,   960,
     963,   964,   970,   971,   979,   980,   983,   985,   989,   992,
     993,   999,  1001,  1005,  1007,  1009,  1011,  1012,  1015,  1017,
    1019,  1021,  1023,  1024,  1032,  1038,  1043,  1044,  1048,  1052,
    1054,  1057,  1061,  1066,  1067,  1076,  1079,  1082,  1083,  1086,
    1088,  1090,  1092,  1094,  1095,  1100,  1102,  1106,  1110,  1112,
    1115,  1119,  1123,  1125,  1127,  1129,  1131,  1133,  1135,  1137,
    1139,  1141,  1144,  1149,  1155,  1159,  1161,  1163,  1167,  1170,
    1174,  1178,  1182,  1186,  1190,  1194,  1198,  1202,  1206,  1210,
    1213,  1218,  1223,  1227,  1230,  1233,  1236,  1239,  1243,  1247,
    1251,  1255,  1259,  1263,  1267,  1271,  1275,  1278,  1282,  1286,
    1290,  1294,  1298,  1301,  1304,  1307,  1309,  1311,  1315,  1320,
    1326,  1329,  1331,  1333,  1335,  1337,  1343,  1347,  1352,  1357,
    1363,  1370,  1375,  1376,  1385,  1386,  1388,  1390,  1393,  1394,
    1401,  1408,  1409,  1418,  1421,  1422,  1428,  1430,  1433,  1439,
    1445,  1450,  1454,  1457,  1462,  1463,  1470,  1474,  1479,  1480,
    1487,  1489,  1493,  1495,  1500,  1502,  1506,  1510
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     108,     0,    -1,   109,    -1,    -1,   109,   110,    -1,   111,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   112,    -1,
     201,    -1,   224,    -1,   242,    -1,   113,    -1,   216,    -1,
     217,    -1,   219,    -1,    39,     6,     3,    -1,    39,     7,
       3,    -1,    39,     1,     3,    -1,   115,    -1,     3,    -1,
      46,     1,     3,    -1,    34,     1,     3,    -1,    32,     1,
       3,    -1,     1,     3,    -1,   253,    83,   256,    -1,   114,
      70,   253,    83,   256,    -1,   256,     3,    -1,   116,    -1,
     117,    -1,   118,    -1,   130,    -1,   147,    -1,   151,    -1,
     164,    -1,   179,    -1,   134,    -1,   145,    -1,   146,    -1,
     190,    -1,   191,    -1,   200,    -1,   251,    -1,   247,    -1,
     214,    -1,   215,    -1,   155,    -1,   156,    -1,   157,    -1,
      10,   114,     3,    -1,    10,     1,     3,    -1,   255,    83,
     256,     3,    -1,   255,    83,   106,   106,     3,    -1,   255,
      83,   274,     3,    -1,   255,    83,   262,     3,    -1,   255,
      70,   276,    83,   256,     3,    -1,   255,    70,   276,    83,
     274,     3,    -1,   119,    -1,   120,    -1,   121,    -1,   122,
      -1,   123,    -1,   124,    -1,   125,    -1,   126,    -1,   127,
      -1,   128,    -1,   129,    -1,   256,    66,   256,     3,    -1,
     255,    65,   256,     3,    -1,   255,    64,   256,     3,    -1,
     255,    63,   256,     3,    -1,   255,    62,   256,     3,    -1,
     255,    56,   256,     3,    -1,   255,    61,   256,     3,    -1,
     255,    60,   256,     3,    -1,   255,    59,   256,     3,    -1,
     255,    57,   256,     3,    -1,   255,    58,   256,     3,    -1,
      -1,   132,   131,   144,     9,     3,    -1,   133,   113,    -1,
      11,   256,     3,    -1,    49,    -1,    11,     1,     3,    -1,
      11,   256,    45,    -1,    49,    45,    -1,    11,     1,    45,
      -1,    -1,   136,   135,   144,   138,     9,     3,    -1,   137,
     113,    -1,    15,   256,     3,    -1,    15,     1,     3,    -1,
      15,   256,    45,    -1,    15,     1,    45,    -1,    -1,   141,
      -1,    -1,   140,   139,   144,    -1,    16,     3,    -1,    16,
       1,     3,    -1,    -1,   143,   142,   144,   138,    -1,    17,
     256,     3,    -1,    17,     1,     3,    -1,    -1,   144,   113,
      -1,    12,     3,    -1,    12,     1,     3,    -1,    13,     3,
      -1,    13,    14,     3,    -1,    13,     1,     3,    -1,    -1,
     149,   148,   144,     9,     3,    -1,   150,   113,    -1,    18,
     255,    83,   256,    69,   256,     3,    -1,    18,   255,    83,
     256,    69,   256,    68,   256,     3,    -1,    18,     1,     3,
      -1,    18,   255,    83,   256,    69,   256,    45,    -1,    18,
     255,    83,   256,    69,   256,    68,   256,    45,    -1,    18,
       1,    45,    -1,    -1,    18,   276,    86,   256,     3,   152,
     153,     9,     3,    -1,    18,   276,    86,   256,    45,   113,
      -1,    18,   276,    86,     1,     3,    -1,    -1,   154,   153,
      -1,   113,    -1,   158,    -1,   160,    -1,   162,    -1,    48,
     256,     3,    -1,    48,     1,     3,    -1,    78,   274,     3,
      -1,    78,     3,    -1,    79,   274,     3,    -1,    79,     3,
      -1,    78,     1,     3,    -1,    50,    -1,    -1,    19,     3,
     159,   144,     9,     3,    -1,    19,    45,   113,    -1,    19,
       1,     3,    -1,    -1,    20,     3,   161,   144,     9,     3,
      -1,    20,    45,   113,    -1,    20,     1,     3,    -1,    -1,
      21,     3,   163,   144,     9,     3,    -1,    21,    45,   113,
      -1,    21,     1,     3,    -1,    -1,   166,   165,   167,   173,
       9,     3,    -1,    22,   256,     3,    -1,    22,     1,     3,
      -1,    -1,   167,   168,    -1,   167,     1,     3,    -1,     3,
      -1,    -1,    23,   177,     3,   169,   144,    -1,    -1,    23,
     177,    45,   170,   113,    -1,    -1,    23,     1,     3,   171,
     144,    -1,    -1,    23,     1,    45,   172,   113,    -1,    -1,
      -1,   175,   174,   176,    -1,    -1,    24,    -1,    24,     1,
      -1,     3,   144,    -1,    45,   113,    -1,   178,    -1,   177,
      70,   178,    -1,     8,    -1,     4,    -1,     7,    -1,     4,
      69,     4,    -1,     6,    -1,    -1,   181,   180,   182,   173,
       9,     3,    -1,    25,   256,     3,    -1,    25,     1,     3,
      -1,    -1,   182,   183,    -1,   182,     1,     3,    -1,     3,
      -1,    -1,    23,   188,     3,   184,   144,    -1,    -1,    23,
     188,    45,   185,   113,    -1,    -1,    23,     1,     3,   186,
     144,    -1,    -1,    23,     1,    45,   187,   113,    -1,   189,
      -1,   188,    70,   189,    -1,    -1,     4,    -1,     6,    -1,
      28,   274,    69,   256,     3,    -1,    28,   274,     1,     3,
      -1,    28,     1,     3,    -1,    29,    45,   113,    -1,    -1,
     193,   192,   144,   194,     9,     3,    -1,    29,     3,    -1,
      29,     1,     3,    -1,    -1,   195,    -1,   196,    -1,   195,
     196,    -1,   197,   144,    -1,    30,     3,    -1,    30,    86,
     253,     3,    -1,    30,   198,     3,    -1,    30,   198,    86,
     253,     3,    -1,    30,     1,     3,    -1,   199,    -1,   198,
      70,   199,    -1,     4,    -1,     6,    -1,    31,   256,     3,
      -1,    31,     1,     3,    -1,   202,   209,   144,     9,     3,
      -1,   204,   113,    -1,   206,    52,   260,   207,   260,    51,
       3,    -1,    -1,   206,    52,   260,   207,     1,   203,   260,
      51,     3,    -1,   206,     1,     3,    -1,   206,    52,   260,
     207,   260,    51,    45,    -1,    -1,   206,    52,   260,     1,
     205,   260,    51,    45,    -1,    46,     6,    -1,    -1,   208,
      -1,   207,    70,   260,   208,    -1,     6,    -1,    -1,    -1,
     212,   210,   144,     9,     3,    -1,    -1,   213,   211,   113,
      -1,    47,     3,    -1,    47,     1,     3,    -1,    47,    45,
      -1,    47,     1,    45,    -1,    40,   258,     3,    -1,    40,
       1,     3,    -1,    43,   256,     3,    -1,    43,   256,    86,
     256,     3,    -1,    43,   256,    86,     1,     3,    -1,    43,
       1,     3,    -1,    41,     6,    83,   252,     3,    -1,    41,
       6,    83,     1,     3,    -1,    41,     1,     3,    -1,    44,
       3,    -1,    44,   218,     3,    -1,    44,     1,     3,    -1,
       6,    -1,   218,    70,     6,    -1,   220,   223,     9,     3,
      -1,   221,   222,     3,    -1,    42,     3,    -1,    42,     1,
       3,    -1,    42,    45,    -1,    42,     1,    45,    -1,    -1,
       6,    -1,   222,    70,     6,    -1,   222,     3,    -1,   223,
     222,     3,    -1,     1,     3,    -1,    -1,    32,     6,   225,
     226,   235,   240,     9,     3,    -1,   227,   229,     3,    -1,
       1,     3,    -1,    -1,    52,   260,   207,   260,    51,    -1,
      -1,    52,   260,   207,     1,   228,   260,    51,    -1,    -1,
      33,   230,    -1,   231,    -1,   230,    70,   231,    -1,     6,
     232,    -1,    -1,    52,   260,   233,   260,    51,    -1,   234,
      -1,   233,    70,   234,    -1,   252,    -1,     6,    -1,    27,
      -1,    -1,   235,   236,    -1,     3,    -1,   201,    -1,   239,
      -1,   237,    -1,    -1,    38,     3,   238,   209,   144,     9,
       3,    -1,    47,     6,    83,   256,     3,    -1,     6,    83,
     256,     3,    -1,    -1,    88,   241,     3,    -1,    88,     1,
       3,    -1,     6,    -1,    74,     6,    -1,   241,    70,     6,
      -1,   241,    70,    74,     6,    -1,    -1,    34,     6,   243,
     244,   245,   240,     9,     3,    -1,   229,     3,    -1,     1,
       3,    -1,    -1,   245,   246,    -1,     3,    -1,   201,    -1,
     239,    -1,   237,    -1,    -1,    36,   248,   249,     3,    -1,
     250,    -1,   249,    70,   250,    -1,   249,    70,     1,    -1,
       6,    -1,    35,     3,    -1,    35,   256,     3,    -1,    35,
       1,     3,    -1,     8,    -1,     4,    -1,     5,    -1,     7,
      -1,     6,    -1,   253,    -1,    27,    -1,    26,    -1,   254,
      -1,   255,   257,    -1,   255,    54,   256,    53,    -1,   255,
      54,    98,   256,    53,    -1,   255,    55,     6,    -1,   252,
      -1,   255,    -1,   256,    95,   256,    -1,    94,   256,    -1,
     256,    94,   256,    -1,   256,    98,   256,    -1,   256,    97,
     256,    -1,   256,    96,   256,    -1,   256,    99,   256,    -1,
     256,    93,   256,    -1,   256,    92,   256,    -1,   256,    91,
     256,    -1,   256,   101,   256,    -1,   256,   100,   256,    -1,
     102,   256,    -1,    75,   255,    82,   256,    -1,    75,   255,
      83,   256,    -1,   256,    80,   256,    -1,   256,   105,    -1,
     105,   256,    -1,   256,   104,    -1,   104,   256,    -1,   256,
      81,   256,    -1,   256,    82,   256,    -1,   256,    83,   256,
      -1,   256,    79,   256,    -1,   256,    78,   256,    -1,   256,
      77,   256,    -1,   256,    76,   256,    -1,   256,    73,   256,
      -1,   256,    72,   256,    -1,    74,   256,    -1,   256,    88,
     256,    -1,   256,    87,   256,    -1,   256,    86,   256,    -1,
     256,    85,   256,    -1,   256,    84,     6,    -1,   106,   256,
      -1,    90,   256,    -1,    89,   256,    -1,   266,    -1,   258,
      -1,   258,    55,     6,    -1,   258,    54,   256,    53,    -1,
     258,    54,    98,   256,    53,    -1,   258,   257,    -1,   269,
      -1,   270,    -1,   272,    -1,   257,    -1,    52,   260,   256,
     260,    51,    -1,    54,    45,    53,    -1,    54,   256,    45,
      53,    -1,    54,    45,   256,    53,    -1,    54,   256,    45,
     256,    53,    -1,   256,    52,   260,   275,   260,    51,    -1,
     256,    52,   260,    51,    -1,    -1,   256,    52,   260,   275,
       1,   259,   260,    51,    -1,    -1,   261,    -1,     3,    -1,
     261,     3,    -1,    -1,    46,   263,   264,   209,   144,     9,
      -1,    52,   260,   207,   260,    51,     3,    -1,    -1,    52,
     260,   207,     1,   265,   260,    51,     3,    -1,     1,     3,
      -1,    -1,    37,   267,   268,    67,   256,    -1,   207,    -1,
       1,     3,    -1,   256,    71,   256,    45,   256,    -1,   256,
      71,   256,    45,     1,    -1,   256,    71,   256,     1,    -1,
     256,    71,     1,    -1,    54,    53,    -1,    54,   275,   260,
      53,    -1,    -1,    54,   275,     1,   271,   260,    53,    -1,
      54,    67,    53,    -1,    54,   277,   260,    53,    -1,    -1,
      54,   277,     1,   273,   260,    53,    -1,   256,    -1,   274,
      70,   256,    -1,   256,    -1,   275,    70,   260,   256,    -1,
     253,    -1,   276,    70,   253,    -1,   256,    67,   256,    -1,
     277,    70,   260,   256,    67,   256,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   195,   197,   201,   202,   203,   208,   213,
     218,   223,   228,   233,   234,   235,   239,   245,   251,   259,
     260,   263,   264,   265,   266,   271,   276,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   298,   299,   300,   301,   302,   303,   304,   308,
     310,   316,   320,   324,   328,   332,   337,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   361,   368,
     375,   382,   389,   396,   403,   410,   417,   423,   429,   437,
     437,   451,   459,   460,   461,   465,   466,   467,   471,   471,
     486,   496,   497,   501,   502,   506,   508,   509,   509,   518,
     519,   524,   524,   536,   537,   540,   542,   548,   557,   565,
     575,   584,   592,   592,   606,   622,   626,   630,   638,   642,
     646,   656,   655,   679,   693,   697,   699,   703,   710,   711,
     712,   716,   729,   737,   741,   747,   752,   759,   768,   778,
     778,   792,   800,   804,   804,   817,   825,   829,   829,   843,
     851,   855,   855,   872,   873,   880,   882,   883,   887,   889,
     888,   899,   899,   911,   911,   923,   923,   939,   942,   941,
     954,   955,   956,   959,   960,   966,   967,   971,   980,   992,
    1003,  1014,  1035,  1035,  1052,  1053,  1060,  1062,  1063,  1067,
    1069,  1068,  1079,  1079,  1092,  1092,  1104,  1104,  1122,  1123,
    1126,  1127,  1139,  1160,  1164,  1169,  1177,  1184,  1183,  1202,
    1203,  1206,  1208,  1212,  1213,  1217,  1222,  1240,  1260,  1270,
    1281,  1289,  1290,  1294,  1306,  1329,  1330,  1337,  1347,  1356,
    1357,  1357,  1361,  1365,  1366,  1366,  1373,  1427,  1429,  1430,
    1434,  1449,  1452,  1451,  1463,  1462,  1477,  1478,  1482,  1483,
    1492,  1496,  1504,  1511,  1521,  1527,  1539,  1549,  1554,  1566,
    1575,  1582,  1590,  1595,  1607,  1614,  1624,  1625,  1628,  1629,
    1632,  1634,  1638,  1645,  1646,  1647,  1659,  1658,  1717,  1720,
    1726,  1728,  1729,  1729,  1735,  1737,  1741,  1742,  1746,  1780,
    1782,  1791,  1792,  1796,  1797,  1806,  1809,  1811,  1815,  1816,
    1819,  1837,  1841,  1841,  1875,  1897,  1924,  1926,  1927,  1934,
    1942,  1948,  1954,  1968,  1967,  2031,  2032,  2038,  2040,  2044,
    2045,  2048,  2067,  2076,  2075,  2093,  2094,  2095,  2102,  2118,
    2119,  2120,  2130,  2131,  2132,  2133,  2137,  2155,  2156,  2157,
    2168,  2169,  2174,  2179,  2185,  2194,  2195,  2196,  2197,  2198,
    2199,  2200,  2201,  2202,  2203,  2204,  2205,  2206,  2207,  2208,
    2209,  2211,  2213,  2214,  2215,  2216,  2217,  2218,  2219,  2220,
    2221,  2222,  2223,  2224,  2225,  2226,  2227,  2228,  2229,  2230,
    2231,  2232,  2233,  2234,  2235,  2236,  2237,  2238,  2242,  2246,
    2250,  2254,  2255,  2256,  2257,  2258,  2263,  2266,  2269,  2272,
    2278,  2284,  2289,  2289,  2297,  2299,  2303,  2304,  2309,  2308,
    2351,  2352,  2352,  2356,  2365,  2364,  2408,  2409,  2418,  2423,
    2430,  2437,  2447,  2448,  2452,  2452,  2460,  2461,  2462,  2462,
    2470,  2471,  2475,  2476,  2480,  2487,  2494,  2495
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORALL", "SWITCH",
  "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "LAMBDA", "INIT",
  "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS", "EXPORT", "COLON",
  "FUNCDECL", "STATIC", "FORDOT", "LOOP", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "ASSIGN_POW",
  "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND",
  "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD",
  "ARROW", "FOR_STEP", "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT",
  "LET", "LE", "GE", "LT", "GT", "NEQ", "EEQ", "OP_EQ", "OP_ASSIGN",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP", "VBAR", "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "load_statement", "statement", "assignment_def_list", "base_statement",
  "def_statement", "assignment", "op_assignment", "autoadd", "autosub",
  "automul", "autodiv", "automod", "autopow", "autoband", "autobor",
  "autobxor", "autoshl", "autoshr", "while_statement", "@1", "while_decl",
  "while_short_decl", "if_statement", "@2", "if_decl", "if_short_decl",
  "elif_or_else", "@3", "else_decl", "elif_statement", "@4", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "for_statement", "@5", "for_decl", "for_decl_short", "forin_statement",
  "@6", "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@7", "last_loop_block", "@8", "all_loop_block", "@9",
  "switch_statement", "@10", "switch_decl", "case_list", "case_statement",
  "@11", "@12", "@13", "@14", "default_statement", "@15", "default_decl",
  "default_body", "case_expression_list", "case_element",
  "select_statement", "@16", "select_decl", "selcase_list",
  "selcase_statement", "@17", "@18", "@19", "@20",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@21", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@22", "func_decl_short", "@23", "func_begin", "param_list",
  "param_symbol", "static_block", "@24", "@25", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@26",
  "class_def_inner", "class_param_list", "@27", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@28", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@29", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@30", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@31",
  "opt_eol", "eol_seq", "nameless_func", "@32", "nameless_func_decl_inner",
  "@33", "lambda_expr", "@34", "lambda_expr_inner", "iif_expr",
  "array_decl", "@35", "dict_decl", "@36", "expression_list",
  "par_expression_list", "symbol_list", "expression_pair_list", 0
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
     355,   356,   357,   358,   359,   360,   361
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   107,   108,   109,   109,   110,   110,   110,   111,   111,
     111,   111,   111,   111,   111,   111,   112,   112,   112,   113,
     113,   113,   113,   113,   113,   114,   114,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   115,   116,
     116,   117,   117,   117,   117,   117,   117,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   131,
     130,   130,   132,   132,   132,   133,   133,   133,   135,   134,
     134,   136,   136,   137,   137,   138,   138,   139,   138,   140,
     140,   142,   141,   143,   143,   144,   144,   145,   145,   146,
     146,   146,   148,   147,   147,   149,   149,   149,   150,   150,
     150,   152,   151,   151,   151,   153,   153,   154,   154,   154,
     154,   155,   155,   156,   156,   156,   156,   156,   157,   159,
     158,   158,   158,   161,   160,   160,   160,   163,   162,   162,
     162,   165,   164,   166,   166,   167,   167,   167,   168,   169,
     168,   170,   168,   171,   168,   172,   168,   173,   174,   173,
     175,   175,   175,   176,   176,   177,   177,   178,   178,   178,
     178,   178,   180,   179,   181,   181,   182,   182,   182,   183,
     184,   183,   185,   183,   186,   183,   187,   183,   188,   188,
     189,   189,   189,   190,   190,   190,   191,   192,   191,   193,
     193,   194,   194,   195,   195,   196,   197,   197,   197,   197,
     197,   198,   198,   199,   199,   200,   200,   201,   201,   202,
     203,   202,   202,   204,   205,   204,   206,   207,   207,   207,
     208,   209,   210,   209,   211,   209,   212,   212,   213,   213,
     214,   214,   215,   215,   215,   215,   216,   216,   216,   217,
     217,   217,   218,   218,   219,   219,   220,   220,   221,   221,
     222,   222,   222,   223,   223,   223,   225,   224,   226,   226,
     227,   227,   228,   227,   229,   229,   230,   230,   231,   232,
     232,   233,   233,   234,   234,   234,   235,   235,   236,   236,
     236,   236,   238,   237,   239,   239,   240,   240,   240,   241,
     241,   241,   241,   243,   242,   244,   244,   245,   245,   246,
     246,   246,   246,   248,   247,   249,   249,   249,   250,   251,
     251,   251,   252,   252,   252,   252,   253,   254,   254,   254,
     255,   255,   255,   255,   255,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   257,   257,   257,   257,
     258,   258,   259,   258,   260,   260,   261,   261,   263,   262,
     264,   265,   264,   264,   267,   266,   268,   268,   269,   269,
     269,   269,   270,   270,   271,   270,   272,   272,   273,   272,
     274,   274,   275,   275,   276,   276,   277,   277
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
       1,     3,     3,     3,     2,     3,     5,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       3,     4,     5,     4,     4,     6,     6,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     0,
       5,     2,     3,     1,     3,     3,     2,     3,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     0,     5,     2,     7,     9,     3,     7,     9,
       3,     0,     9,     6,     5,     0,     2,     1,     1,     1,
       1,     3,     3,     3,     2,     3,     2,     3,     1,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     0,     0,     3,
       0,     1,     2,     2,     2,     1,     3,     1,     1,     1,
       3,     1,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     1,     3,
       0,     1,     1,     5,     4,     3,     3,     0,     6,     2,
       3,     0,     1,     1,     2,     2,     2,     4,     3,     5,
       3,     1,     3,     1,     1,     3,     3,     5,     2,     7,
       0,     9,     3,     7,     0,     8,     2,     0,     1,     4,
       1,     0,     0,     5,     0,     3,     2,     3,     2,     3,
       3,     3,     3,     5,     5,     3,     5,     5,     3,     2,
       3,     3,     1,     3,     4,     3,     2,     3,     2,     3,
       0,     1,     3,     2,     3,     2,     0,     8,     3,     2,
       0,     5,     0,     7,     0,     2,     1,     3,     2,     0,
       5,     1,     3,     1,     1,     1,     0,     2,     1,     1,
       1,     1,     0,     7,     5,     4,     0,     3,     3,     1,
       2,     3,     4,     0,     8,     2,     2,     0,     2,     1,
       1,     1,     1,     0,     4,     1,     3,     3,     1,     2,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     4,     5,     3,     1,     1,     3,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     2,
       4,     4,     3,     2,     2,     2,     2,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     3,     3,     3,
       3,     3,     2,     2,     2,     1,     1,     3,     4,     5,
       2,     1,     1,     1,     1,     5,     3,     4,     4,     5,
       6,     4,     0,     8,     0,     1,     1,     2,     0,     6,
       6,     0,     8,     2,     0,     5,     1,     2,     5,     5,
       4,     3,     2,     4,     0,     6,     3,     4,     0,     6,
       1,     3,     1,     4,     1,     3,     3,     6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    20,   333,   334,   336,   335,
     332,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   339,   338,     0,     0,     0,     0,     0,     0,   323,
     414,     0,     0,     0,     0,     0,     0,     0,     0,    83,
     138,   404,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    12,    19,    28,
      29,    30,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    31,    79,     0,    36,    88,     0,    37,
      38,    32,   112,     0,    33,    46,    47,    48,    34,   151,
      35,   182,    39,    40,   207,    41,     9,   241,     0,     0,
      44,    45,    13,    14,    15,     0,   270,    10,    11,    43,
      42,   345,   337,   340,   346,     0,   394,   386,   385,   391,
     392,   393,    24,     6,     0,     0,     0,     0,   346,     0,
       0,   107,     0,   109,     0,     0,     0,     0,   337,     0,
       0,     0,     0,     0,     0,     0,     0,   430,     0,     0,
     209,     0,     0,     0,     0,   276,     0,   313,     0,   329,
       0,     0,     0,     0,     0,     0,     0,     0,   386,     0,
       0,     0,   266,   268,     0,     0,     0,   259,   262,     0,
       0,   236,     0,     0,    86,   406,     0,   405,     0,   422,
       0,   432,     0,     0,   376,     0,     0,   134,     0,   136,
       0,   384,   383,   348,   359,   366,   364,   382,   105,     0,
       0,     0,    81,   105,    90,   105,   114,   155,   186,   105,
       0,   105,   242,   244,   228,     0,   404,     0,   271,     0,
     270,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   341,    27,   404,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   365,   363,     0,
       0,   390,    50,    49,     0,     0,    84,    87,    82,    85,
     108,   111,   110,    92,    94,    91,    93,   117,   120,     0,
       0,     0,   154,   153,     7,   185,   184,   205,     0,     0,
       0,   210,   206,   226,   225,    23,     0,    22,     0,   331,
     330,   328,     0,   325,     0,   240,   416,   238,     0,    18,
      16,    17,   251,   250,   258,     0,   267,   269,   255,   252,
       0,   261,   260,     0,    21,   132,   131,   404,   407,   396,
       0,   426,     0,     0,   424,   404,     0,   428,   404,     0,
       0,     0,   137,   133,   135,     0,     0,     0,     0,     0,
       0,     0,   246,   248,     0,   105,     0,   232,     0,   275,
     273,     0,     0,     0,   265,     0,     0,   344,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   434,     0,
     408,     0,   430,     0,     0,     0,     0,   421,     0,   375,
     374,   373,   372,   371,   370,   362,   367,   368,   369,   381,
     380,   379,   378,   377,   356,   355,   354,   349,   347,   352,
     351,   350,   353,   358,   357,     0,     0,   387,     0,    25,
       0,   435,     0,     0,   204,     0,   431,     0,   404,   296,
     284,     0,     0,     0,   317,   324,     0,   417,   404,     0,
       0,     0,     0,   379,   263,     0,   398,   397,     0,   436,
     404,     0,   423,   404,     0,   427,   360,   361,     0,   106,
       0,     0,     0,    97,    96,   101,     0,     0,   158,     0,
       0,   156,     0,   168,     0,   189,     0,     0,   187,     0,
       0,   212,   213,   105,   247,   249,     0,     0,   245,   234,
       0,   272,   264,   274,     0,   342,    73,    77,    78,    76,
      75,    74,    72,    71,    70,    69,     0,     0,     0,    51,
      54,    53,   401,   432,     0,    68,   420,     0,     0,   388,
       0,     0,   124,   121,     0,   203,   279,   237,   306,     0,
     316,   289,   285,   286,   315,   306,   327,   326,     0,   415,
     257,   256,   254,   253,   395,   399,     0,   433,     0,     0,
      80,     0,    99,     0,     0,     0,   105,   105,   113,   157,
       0,   178,   181,   179,   177,     0,   175,   172,     0,     0,
     188,     0,   201,   202,     0,   198,     0,     0,   216,   223,
     224,     0,     0,   221,     0,   214,     0,   227,     0,   404,
     230,     0,   343,   430,     0,     0,   404,   241,    52,   402,
       0,   419,   418,   389,    26,     0,     0,   123,     0,   298,
       0,     0,     0,     0,     0,   299,   297,   301,   300,     0,
     278,   404,   288,     0,   319,   320,   322,   321,     0,   318,
     239,   425,   429,     0,   100,   104,   103,    89,     0,     0,
     163,   165,     0,   159,   161,     0,   152,   105,     0,   169,
     194,   196,   190,   192,   200,   183,   220,     0,   218,     0,
       0,   208,   243,     0,   404,     0,    55,    56,   413,   237,
     105,   404,   400,   115,   118,     0,     0,     0,     0,   127,
       0,     0,   128,   129,   130,   282,     0,     0,   302,     0,
       0,   309,     0,     0,     0,     0,   287,     0,   437,   102,
     105,     0,   180,   105,     0,   176,     0,   174,   105,     0,
     105,     0,   199,   217,   222,     0,     0,     0,   229,   233,
       0,     0,     0,     0,     0,   139,     0,     0,   143,     0,
       0,   147,     0,     0,   126,   404,   281,     0,   241,     0,
     308,   310,   307,     0,   277,   294,   295,   404,   291,   293,
     314,     0,   166,     0,   162,     0,   197,     0,   193,   219,
     235,     0,   411,     0,   409,   403,   116,   119,   142,   105,
     141,   146,   105,   145,   150,   105,   149,   122,     0,   305,
     105,     0,   311,     0,     0,     0,   231,   404,     0,     0,
       0,     0,   283,     0,   304,   312,   292,   290,     0,   410,
       0,     0,     0,     0,     0,   140,   144,   148,   303,   412
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    54,    55,    56,   479,   125,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,   208,    74,    75,    76,   213,    77,
      78,   482,   576,   483,   484,   577,   485,   365,    79,    80,
      81,   215,    82,    83,    84,   626,   700,   701,    85,    86,
      87,   702,   789,   703,   792,   704,   795,    88,   217,    89,
     368,   491,   723,   724,   720,   721,   492,   589,   493,   669,
     585,   586,    90,   218,    91,   369,   498,   730,   731,   728,
     729,   594,   595,    92,    93,   219,    94,   500,   501,   502,
     503,   602,   603,    95,    96,    97,   684,    98,   609,    99,
     326,   327,   221,   375,   376,   222,   223,   100,   101,   102,
     103,   179,   104,   105,   106,   229,   230,   107,   316,   449,
     450,   755,   453,   552,   553,   642,   767,   768,   548,   636,
     637,   758,   638,   639,   713,   108,   318,   454,   555,   649,
     109,   161,   322,   323,   110,   111,   112,   113,   128,   115,
     116,   117,   691,   186,   187,   403,   527,   617,   807,   118,
     162,   328,   119,   120,   470,   121,   473,   148,   192,   140,
     193
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -616
static const yytype_int16 yypact[] =
{
    -616,    84,   812,  -616,    65,  -616,  -616,  -616,  -616,  -616,
    -616,    92,   190,   128,   528,   581,   330,   574,   450,     8,
     559,  -616,  -616,   688,   182,  2987,   213,   327,    13,  -616,
    -616,   406,  3042,   440,   398,  3097,   204,   492,  3152,    76,
    -616,   150,  5031,  5312,   532,   119,  3569,  5312,  5312,  5312,
    5312,  5312,  5312,  5312,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  2932,  -616,  -616,  2932,  -616,
    -616,  -616,  -616,  2932,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,   131,  2932,    23,
    -616,  -616,  -616,  -616,  -616,    42,   178,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,   919,  3652,  -616,   271,  -616,  -616,
    -616,  -616,  -616,  -616,   189,    46,   117,   154,   361,  3714,
     284,  -616,   293,  -616,   300,   267,  3771,   303,   -40,   348,
      15,   304,  3935,   329,   337,  3972,   373,  5768,    82,   377,
    -616,  2932,   420,  4022,   426,  -616,   430,  -616,   436,  -616,
    4059,   442,    93,   439,   457,   471,   475,  5768,   324,   489,
     443,   336,  -616,  -616,   519,  4109,   529,  -616,  -616,    68,
     544,  -616,   545,  4146,  -616,  -616,  5312,   554,  5162,  -616,
     508,  5382,    89,   136,  5879,   260,   565,  -616,   106,  -616,
     145,   577,   577,   -26,   -26,   -26,   -26,   -26,  -616,   561,
     568,   572,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
     405,  -616,  -616,  -616,  -616,   573,   150,   588,  -616,   158,
     349,   183,  5055,   571,  5312,  5312,  5312,  5312,  5312,  5312,
    5312,  5312,  5312,  5312,   591,  5196,  -616,  -616,   150,  5312,
    3207,  5312,  5312,  5312,  5312,  5312,  5312,  5312,  5312,  5312,
    5312,   593,  5312,  5312,  5312,  5312,  5312,  5312,  5312,  5312,
    5312,  5312,  5312,  5312,  5312,  5312,  5312,  -616,  -616,  5089,
     596,  -616,  -616,  -616,   591,  5312,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  5312,
     591,  3262,  -616,  -616,  -616,  -616,  -616,  -616,   600,  5312,
    5312,  -616,  -616,  -616,  -616,  -616,   397,  -616,   319,  -616,
    -616,  -616,   234,  -616,   613,  -616,   524,  -616,   553,  -616,
    -616,  -616,  -616,  -616,  -616,   614,  -616,  -616,  -616,  -616,
    3317,  -616,  -616,   611,  -616,  -616,  -616,  4196,  -616,  -616,
    5533,  -616,  5220,  5312,  -616,   150,   570,  -616,   150,   578,
    5312,  5312,  -616,  -616,  -616,  1766,  1448,  1872,   472,   569,
    1554,   392,  -616,  -616,  1978,  -616,  2932,  -616,   107,  -616,
    -616,   620,   624,   235,  -616,  5312,  5439,  -616,  4233,  4283,
    4320,  4370,  4407,  4457,  4494,  4544,  4581,  4631,  -616,    58,
    -616,  5346,  4668,   627,   253,  5254,  4718,  -616,  3466,  5842,
    5879,   784,   784,   784,   784,   784,   784,   784,   784,  -616,
     577,   577,   577,   577,   292,   292,   660,   365,   365,   246,
     246,   246,   321,   -26,   -26,  5312,  5496,  -616,   558,  5768,
    5570,  -616,   633,  3828,  -616,  4755,  5768,   639,   150,  -616,
     612,   641,   646,   644,  -616,  -616,   522,  -616,   150,  5312,
     651,   652,   655,   415,  -616,   608,  -616,  -616,  5607,  5768,
     150,  5312,  -616,   150,  5312,  -616,   784,   784,   657,  -616,
     550,  3372,   670,  -616,  -616,  -616,   677,   680,  -616,   631,
     408,  -616,   675,  -616,   682,  -616,   296,   678,  -616,     7,
     679,   656,  -616,  -616,  -616,  -616,   687,  2084,  -616,  -616,
     161,  -616,  -616,  -616,  5644,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  5312,    52,  3575,  -616,
    -616,  -616,  -616,  5768,   165,  -616,  -616,  3427,  5681,  -616,
    5312,  5312,  -616,  -616,  2932,  -616,  -616,   691,   141,   695,
    -616,   649,   632,  -616,  -616,   283,  -616,  -616,   691,  5768,
    -616,  -616,  -616,  -616,  -616,  -616,   650,  5768,   653,  5731,
    -616,   702,  -616,   704,  4805,   705,  -616,  -616,  -616,  -616,
     476,   640,  -616,  -616,  -616,    60,  -616,  -616,   707,   488,
    -616,   491,  -616,  -616,   166,  -616,   708,   710,  -616,  -616,
    -616,   591,    28,  -616,   714,  -616,  1660,  -616,   715,   150,
    -616,   681,  -616,  4842,   254,   718,   150,   131,  -616,  -616,
     683,  -616,  5805,  -616,  5768,  3615,   918,  -616,   169,  -616,
     643,   728,   727,   729,    32,  -616,  -616,  -616,  -616,   732,
    -616,   150,  -616,   646,  -616,  -616,  -616,  -616,   734,  -616,
    -616,  -616,  -616,  5312,  -616,  -616,  -616,  -616,  2190,  1448,
    -616,  -616,   733,  -616,  -616,   493,  -616,  -616,  2932,  -616,
    -616,  -616,  -616,  -616,   583,  -616,  -616,   735,  -616,   584,
     591,  -616,  -616,   685,   150,   500,  -616,  -616,  -616,   691,
    -616,   150,  -616,  -616,  -616,  5312,   437,   444,   485,  -616,
     737,   918,  -616,  -616,  -616,  -616,   698,  5312,  -616,   669,
     750,  -616,   760,   289,   764,   601,  -616,   767,  5768,  -616,
    -616,  2932,  -616,  -616,  2932,  -616,  2296,  -616,  -616,  2932,
    -616,  2932,  -616,  -616,  -616,   769,   731,   723,  -616,  -616,
     239,  2402,   730,  3885,   776,  -616,  2932,   777,  -616,  2932,
     780,  -616,  2932,   781,  -616,   150,  -616,  4892,   131,  5312,
    -616,  -616,  -616,     6,  -616,  -616,  -616,   290,  -616,  -616,
    -616,  1024,  -616,  1130,  -616,  1236,  -616,  1342,  -616,  -616,
    -616,   782,  -616,   736,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,   738,  -616,
    -616,  4929,  -616,   785,   601,   744,  -616,   150,   783,  2508,
    2614,  2720,  -616,  2826,  -616,  -616,  -616,  -616,   745,  -616,
     794,   795,   796,   797,   798,  -616,  -616,  -616,  -616,  -616
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -616,  -616,  -616,  -616,  -616,  -616,    -1,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,   129,  -616,  -616,  -616,  -616,  -616,  -149,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,   101,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,   434,  -616,  -616,  -616,
    -616,   139,  -616,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,   132,  -616,  -616,  -616,  -616,  -616,  -616,   309,
    -616,  -616,   126,  -616,  -374,  -616,  -616,  -616,  -616,  -616,
    -372,   256,  -615,  -616,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,   -99,  -616,  -616,  -616,  -616,
    -616,  -616,   376,  -616,   185,  -616,  -616,    25,  -616,  -616,
     276,  -616,   277,   278,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,   386,  -616,  -332,    11,  -616,    -2,     9,
     -28,   813,  -616,  -157,  -616,  -616,  -616,  -616,  -616,  -616,
    -616,  -616,  -616,  -616,  -616,  -616,  -616,   -41,   445,   615,
    -616
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -435
static const yytype_int16 yytable[] =
{
     114,    57,   690,   461,   198,   200,   510,   231,   597,   143,
     598,   599,   802,   600,   158,   139,   159,     6,     7,     8,
       9,    10,   129,   126,   225,   136,   248,   142,   138,   145,
    -434,   678,   147,   710,   153,   356,   359,   160,   711,    21,
      22,   167,   195,   227,   175,  -270,  -434,   183,   228,   283,
      30,   191,   194,   615,   147,   147,   201,   202,   203,   204,
     205,   206,   207,   663,   366,    41,   367,    42,   122,   378,
     370,   342,   374,   114,   212,   226,   114,   214,   277,   278,
     803,   114,   216,   308,     3,   300,   246,    43,    44,   281,
     354,   405,   185,   601,   324,   123,   114,   224,   679,   325,
     246,   301,    47,    48,   616,   664,   712,    49,   509,   363,
    -237,   246,  -270,   325,   680,    50,   284,    51,    52,    53,
     196,   184,   197,     6,     7,     8,     9,    10,   300,   127,
     665,   383,     6,     7,     8,     9,    10,   357,   343,   185,
     281,   526,  -404,   800,   629,    21,    22,   630,   364,   114,
     312,   309,   310,   185,    21,    22,    30,   286,  -237,   355,
    -237,   380,   610,  -237,   185,    30,   619,   246,   185,   672,
     705,    41,   185,    42,   635,   628,   310,  -237,   220,   631,
      41,   645,    42,   149,   228,   150,   384,   632,   633,  -404,
     465,   124,   282,    43,    44,   347,     8,   350,   471,   287,
     285,   474,    43,    44,   404,   176,   358,   177,    47,    48,
     178,   673,  -404,    49,   154,   310,  -404,    47,    48,   155,
    -404,    50,    49,    51,    52,    53,   507,   151,   381,   634,
      50,   458,    51,    52,    53,   355,   674,   455,   513,   458,
     782,   386,   185,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   381,   402,   398,   531,   687,   406,   408,
     409,   410,   411,   412,   413,   414,   415,   416,   417,   418,
     293,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   644,   290,   436,   630,
    -404,   547,   762,   185,   439,   438,   291,   591,   248,  -200,
     592,   558,   593,   292,   456,   381,   297,   302,   440,   458,
     443,   441,   294,   566,   232,   233,   568,   740,   445,   446,
     451,   631,  -284,   310,   310,   279,   280,   333,   156,   632,
     633,   135,   304,   157,     6,     7,     8,     9,    10,   336,
     305,  -200,   360,   361,   248,   274,   275,   276,   298,   463,
     277,   278,   452,   611,   606,   228,    21,    22,   382,   763,
     804,   468,   469,   114,   114,   114,  -200,    30,   114,   476,
     477,   634,   114,   248,   114,   508,   307,   620,   279,   280,
     311,   337,    41,   769,    42,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   514,   504,   277,   278,   447,   171,
    -280,   172,   232,   233,    43,    44,   371,   163,   372,   587,
     207,  -171,   164,   165,   533,   232,   233,   248,   563,    47,
      48,   275,   276,   313,    49,   277,   278,   658,   659,   315,
    -280,   299,    50,   317,    51,    52,    53,   505,   744,   319,
     745,   169,   329,   173,   538,   747,   170,   748,   321,   448,
     373,   141,   683,  -171,     6,     7,     8,     9,    10,   689,
     330,   271,   272,   273,   274,   275,   276,   248,   559,   277,
     278,   706,   769,   487,   331,   488,    21,    22,   332,   660,
     567,  -167,   746,   569,   715,   614,   750,    30,   751,   749,
     574,   667,   334,   180,   670,   489,   490,   581,   181,   582,
     583,   584,    41,   738,    42,   114,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,  -170,   726,   277,
     278,   661,   338,   556,    43,    44,   335,   737,   321,   130,
     752,   131,   341,   668,   742,   613,   671,   207,     8,    47,
      48,   741,   114,   627,    49,   739,   622,   344,   345,   624,
     625,   571,    50,   572,    51,    52,    53,   348,    21,    22,
     144,   351,   154,     6,     7,     8,     9,    10,   362,   156,
     494,   771,   495,   180,   773,   137,   377,   387,  -167,   775,
       8,   777,   132,   783,   133,    21,    22,   592,   599,   593,
     600,   379,   496,   490,   458,   134,    30,     8,   798,   419,
      21,    22,   437,   444,   114,     6,     7,   765,     9,    10,
     805,    41,   677,    42,  -170,   460,   457,   464,     6,     7,
     459,     9,    10,   472,   114,   699,   511,   512,   766,   248,
     530,   475,   580,    43,    44,   581,   542,   582,   583,   584,
     809,   540,   546,   810,   550,   452,   811,   554,    47,    48,
     818,   813,   551,    49,   560,   561,   114,   114,   562,   564,
     570,    50,   718,    51,    52,    53,   114,   727,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   575,
     578,   277,   278,   579,   588,   590,   499,   596,   604,   146,
     607,   735,     6,     7,     8,     9,    10,   325,   640,   114,
     699,   641,   643,   651,   743,   654,   652,   655,   657,   662,
     666,   675,   248,   676,    21,    22,   757,   681,   682,   114,
     772,   688,   114,   774,   114,    30,   707,   114,   776,   114,
     778,   708,   685,   181,   692,   709,   736,   722,   733,   114,
      41,   714,    42,   717,   114,   790,   753,   114,   793,   756,
     114,   796,   759,   760,   269,   270,   271,   272,   273,   274,
     275,   276,    43,    44,   277,   278,   761,   764,   801,   114,
     770,   114,   779,   114,   781,   114,   780,    47,    48,   788,
     791,   785,    49,   794,   797,   806,   819,   808,   719,   812,
      50,   815,    51,    52,    53,   817,   824,   825,   826,   827,
     828,   829,   754,   497,   725,   734,   732,   114,   114,   114,
     605,   114,    -2,     4,   650,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,   549,    16,   716,   816,
      17,   646,   647,   648,    18,    19,   248,    20,    21,    22,
      23,    24,   557,    25,    26,   168,    27,    28,    29,    30,
     534,    31,    32,    33,    34,    35,    36,     0,    37,   399,
      38,    39,    40,     0,    41,     0,    42,     0,   261,   262,
     263,   264,   265,     0,     0,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,    43,    44,   277,   278,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,  -125,    12,    13,
      14,    15,     0,    16,     0,     0,    17,   696,   697,   698,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     209,     0,   210,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   211,     0,    38,    39,    40,     0,
      41,     0,    42,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,     0,     0,     0,     0,   244,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,   245,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,  -164,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -164,  -164,    20,
      21,    22,    23,    24,     0,    25,   209,     0,   210,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,  -164,
     211,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,  -160,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -160,  -160,    20,    21,    22,    23,    24,
       0,    25,   209,     0,   210,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,  -160,   211,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,  -195,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -195,
    -195,    20,    21,    22,    23,    24,     0,    25,   209,     0,
     210,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,  -195,   211,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,  -191,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -191,  -191,    20,    21,    22,
      23,    24,     0,    25,   209,     0,   210,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,  -191,   211,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,   -95,    12,    13,
      14,    15,     0,    16,   480,   481,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     209,     0,   210,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   211,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,  -211,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,   499,    25,   209,     0,   210,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
     211,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,  -215,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    -215,    25,   209,     0,   210,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   211,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,   478,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   209,     0,
     210,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   211,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,   486,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   209,     0,   210,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   211,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,   506,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     209,     0,   210,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   211,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,   608,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   209,     0,   210,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
     211,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,   -98,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   209,     0,   210,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   211,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,  -173,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   209,     0,
     210,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   211,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,   784,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   209,     0,   210,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   211,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,   820,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     209,     0,   210,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   211,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,   821,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   209,     0,   210,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
     211,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,   822,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   209,     0,   210,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   211,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,   823,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   209,     0,
     210,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   211,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,     0,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   209,     0,   210,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   211,     0,
      38,    39,    40,     0,    41,     0,    42,     0,   152,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,    21,    22,     0,     0,     0,     0,     0,
       0,    47,    48,     0,    30,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,    41,
       0,    42,     0,   166,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,    47,    48,     0,    30,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,    41,     0,    42,     0,   174,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,    47,    48,     0,    30,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,    41,
       0,    42,     0,   182,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,    47,    48,     0,    30,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,    41,     0,    42,     0,   407,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,    47,    48,     0,    30,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,    41,
       0,    42,     0,   442,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,    47,    48,     0,    30,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,    41,     0,    42,     0,   462,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,    47,    48,     0,    30,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,    41,
       0,    42,     0,   573,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,    47,    48,     0,    30,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,    41,     0,    42,     0,   621,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,    47,    48,     0,    30,     0,    49,   536,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,     0,     0,     0,     0,
       0,   537,     0,     0,     0,     0,    47,    48,   248,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     0,     0,     0,   250,   251,   252,
       0,     0,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,     0,     0,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,     0,     0,
     277,   278,   199,     6,     7,     8,     9,    10,   618,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    30,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,   693,     0,
       0,    41,     0,    42,     0,     0,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,   247,     0,     0,    47,    48,
     694,     0,     0,    49,    47,    48,     0,   248,     0,    49,
       0,    50,     0,    51,    52,    53,     0,    50,     0,    51,
      52,    53,     0,   695,     0,     0,   250,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   248,     0,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   288,   249,   277,
     278,     0,     0,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,     0,     0,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     0,     0,   277,   278,     0,   289,
       0,     0,     0,     0,     0,     0,   248,     0,     0,     0,
       0,     0,     0,     0,   295,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   250,   251,   252,     0,     0,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,     0,     0,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   296,     0,   277,   278,
       0,     0,     0,   248,     0,     0,     0,     0,     0,     0,
       0,   543,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   250,   251,   252,     0,     0,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
       0,     0,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   544,     0,   277,   278,     0,     0,     0,
     248,     0,     0,     0,     0,     0,     0,     0,   786,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   250,
     251,   252,     0,     0,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,     0,     0,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     787,     0,   277,   278,     0,     0,     0,   248,   303,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   250,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,     0,   306,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   248,     0,   277,
     278,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   250,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   248,   314,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,     0,   277,
     278,     0,     0,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,     0,   320,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   248,     0,   277,   278,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   248,   339,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     0,     0,   277,   278,     0,     0,
     250,   251,   252,     0,     0,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,     0,   346,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   248,     0,   277,   278,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     250,   251,   252,     0,     0,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   340,   264,   265,   248,   185,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,     0,     0,   250,   251,   252,
       0,     0,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,     0,   516,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   248,     0,
     277,   278,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   250,   251,   252,
       0,     0,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   248,   517,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,     0,     0,
     277,   278,     0,     0,   250,   251,   252,     0,     0,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,     0,   518,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   248,     0,   277,   278,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   250,   251,   252,     0,     0,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   248,   519,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,     0,     0,   277,   278,     0,
       0,   250,   251,   252,     0,     0,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,     0,
     520,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   248,     0,   277,   278,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   250,   251,   252,     0,     0,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   248,
     521,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,     0,   277,   278,     0,     0,   250,   251,
     252,     0,     0,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,     0,   522,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   248,
       0,   277,   278,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   250,   251,
     252,     0,     0,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   248,   523,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,     0,
       0,   277,   278,     0,     0,   250,   251,   252,     0,     0,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,     0,   524,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   248,     0,   277,   278,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   250,   251,   252,     0,     0,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   248,   525,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,     0,     0,   277,   278,
       0,     0,   250,   251,   252,     0,     0,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
       0,   529,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   248,     0,   277,   278,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   250,   251,   252,     0,     0,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     248,   535,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,     0,     0,   277,   278,     0,     0,   250,
     251,   252,     0,     0,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,     0,   545,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     248,     0,   277,   278,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   250,
     251,   252,     0,     0,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   248,   656,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
       0,     0,   277,   278,     0,     0,   250,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,     0,   686,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   248,     0,   277,
     278,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   250,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   248,   799,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,     0,   277,
     278,     0,     0,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,     0,   814,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   248,     0,   277,   278,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   248,     0,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     0,     0,   277,   278,     0,     0,
     250,   251,   252,     0,     0,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,     0,     0,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     6,
       7,     8,     9,    10,     0,     0,     0,     0,    30,     0,
       0,     0,     0,     0,     0,     0,   188,     0,     0,     0,
       0,    21,    22,    41,   189,    42,     0,     0,     0,     0,
       0,     0,    30,     6,     7,     8,     9,    10,   190,     0,
     188,     0,     0,     0,     0,    43,    44,    41,     0,    42,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
      47,    48,     0,     0,     0,    49,    30,     0,     0,    43,
      44,     0,     0,    50,   188,    51,    52,    53,     0,     0,
       0,    41,     0,    42,    47,    48,     0,     0,     0,    49,
       0,     0,     0,   385,     0,     0,     0,    50,     0,    51,
      52,    53,     0,    43,    44,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,     0,     0,   435,    21,    22,
       0,    50,     0,    51,    52,    53,     0,     0,     0,    30,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    41,   349,    42,     0,     0,     0,
       0,     0,    21,    22,     6,     7,     8,     9,    10,     0,
       0,     0,     0,    30,     0,     0,    43,    44,     0,     0,
       0,     0,   400,     0,     0,     0,    21,    22,    41,     0,
      42,    47,    48,     0,     0,     0,    49,    30,     6,     7,
       8,     9,    10,     0,    50,     0,    51,    52,    53,     0,
      43,    44,    41,   467,    42,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,    47,    48,     0,     0,     0,
      49,    30,     0,     0,    43,    44,     0,     0,    50,     0,
      51,    52,   401,     0,     0,   532,    41,     0,    42,    47,
      48,     0,     0,     0,    49,     0,     6,     7,     8,     9,
      10,     0,    50,     0,    51,    52,    53,     0,    43,    44,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,    47,    48,     0,     0,     0,    49,    30,
       6,     7,     8,     9,    10,     0,    50,     0,    51,    52,
      53,     0,     0,     0,    41,     0,    42,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    30,     0,     0,    43,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    41,     0,
      42,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     0,
      43,    44,     0,     0,     0,     0,     0,   352,     0,     0,
       0,     0,     0,     0,   248,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,   353,
      51,    52,   528,   250,   251,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,     0,     0,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   352,     0,   277,   278,     0,     0,
       0,   248,   515,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     250,   251,   252,     0,     0,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,     0,     0,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   352,     0,   277,   278,     0,     0,     0,   248,   539,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   250,   251,   252,
       0,     0,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   248,   466,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,     0,     0,
     277,   278,     0,     0,   250,   251,   252,     0,     0,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   248,     0,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,     0,     0,   277,   278,   541,
       0,   250,   251,   252,     0,     0,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   248,
     565,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,     0,   277,   278,     0,     0,   250,   251,
     252,     0,     0,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   248,   612,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,     0,
       0,   277,   278,     0,     0,   250,   251,   252,     0,     0,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   248,   623,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,     0,     0,   277,   278,
       0,     0,   250,   251,   252,     0,     0,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
       0,     0,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   248,     0,   277,   278,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   653,     0,
       0,     0,   250,   251,   252,     0,     0,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     248,     0,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,     0,     0,   277,   278,     0,     0,   250,
     251,   252,     0,     0,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   248,     0,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
       0,     0,   277,   278,     0,     0,     0,   251,   252,     0,
       0,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   248,     0,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,     0,   277,
     278,     0,     0,     0,     0,   252,     0,     0,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   248,     0,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     0,     0,   277,   278,     0,     0,
       0,     0,     0,     0,     0,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,     0,     0,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278
};

static const yytype_int16 yycheck[] =
{
       2,     2,   617,   335,    45,    46,   378,   106,     1,     1,
       3,     4,     6,     6,     1,    17,     3,     4,     5,     6,
       7,     8,    13,    12,     1,    16,    52,    18,    17,    20,
      70,     3,    23,     1,    25,   192,   193,    28,     6,    26,
      27,    32,    44,     1,    35,     3,    86,    38,     6,     3,
      37,    42,    43,     1,    45,    46,    47,    48,    49,    50,
      51,    52,    53,     3,   213,    52,   215,    54,     3,   226,
     219,     3,   221,    75,    75,    52,    78,    78,   104,   105,
      74,    83,    83,     1,     0,    70,   114,    74,    75,   117,
       1,   248,     3,    86,     1,     3,    98,    98,    70,     6,
     128,    86,    89,    90,    52,    45,    74,    94,     1,     3,
       3,   139,    70,     6,    86,   102,    70,   104,   105,   106,
       1,    45,     3,     4,     5,     6,     7,     8,    70,     1,
      70,   230,     4,     5,     6,     7,     8,     1,    70,     3,
     168,    83,    53,   758,     3,    26,    27,     6,     3,   151,
     151,    69,    70,     3,    26,    27,    37,     3,    51,    70,
      67,     3,     1,    70,     3,    37,     1,   195,     3,     3,
       1,    52,     3,    54,   548,   547,    70,    70,    47,    38,
      52,   555,    54,     1,     6,     3,     3,    46,    47,    53,
     347,     1,     3,    74,    75,   186,     6,   188,   355,    45,
      83,   358,    74,    75,   245,     1,    70,     3,    89,    90,
       6,    45,    51,    94,     1,    70,    51,    89,    90,     6,
      51,   102,    94,   104,   105,   106,   375,    45,    70,    88,
     102,    70,   104,   105,   106,    70,    70,     3,     3,    70,
       1,   232,     3,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,    70,   245,   244,     3,     3,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
       3,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,     3,     3,   279,     6,
      51,   448,     3,     3,   285,   284,     3,     1,    52,     3,
       4,   458,     6,     3,    70,    70,     3,     3,   299,    70,
     301,   300,    45,   470,    54,    55,   473,   689,   309,   310,
       1,    38,     3,    70,    70,    54,    55,     3,     1,    46,
      47,     1,     3,     6,     4,     5,     6,     7,     8,     3,
       3,    45,    82,    83,    52,    99,   100,   101,    45,   340,
     104,   105,    33,   510,   503,     6,    26,    27,     9,    70,
      70,   352,   353,   365,   366,   367,    70,    37,   370,   360,
     361,    88,   374,    52,   376,   376,     3,   534,    54,    55,
       3,    45,    52,   715,    54,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   385,     3,   104,   105,     1,     1,
       3,     3,    54,    55,    74,    75,     1,     1,     3,     1,
     401,     3,     6,     7,   405,    54,    55,    52,     3,    89,
      90,   100,   101,     3,    94,   104,   105,   576,   577,     3,
      33,    83,   102,     3,   104,   105,   106,    45,     1,     3,
       3,     1,     3,    45,   435,     1,     6,     3,     6,    52,
      45,     1,   609,    45,     4,     5,     6,     7,     8,   616,
       3,    96,    97,    98,    99,   100,   101,    52,   459,   104,
     105,   628,   804,     1,     3,     3,    26,    27,     3,     3,
     471,     9,    45,   474,   641,   526,     1,    37,     3,    45,
     481,     3,     3,     1,     3,    23,    24,     4,     6,     6,
       7,     8,    52,     3,    54,   507,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    45,   667,   104,
     105,    45,     3,     1,    74,    75,    83,   684,     6,     1,
      45,     3,     3,    45,   691,   526,    45,   528,     6,    89,
      90,   690,   544,   544,    94,    45,   537,     3,     3,   540,
     541,     1,   102,     3,   104,   105,   106,     3,    26,    27,
       1,    53,     1,     4,     5,     6,     7,     8,     3,     1,
       1,   720,     3,     1,   723,     1,     3,     6,     9,   728,
       6,   730,     1,   740,     3,    26,    27,     4,     4,     6,
       6,     3,    23,    24,    70,    14,    37,     6,   755,     6,
      26,    27,     6,     3,   606,     4,     5,     6,     7,     8,
     767,    52,   601,    54,    45,     1,     3,     6,     4,     5,
      67,     7,     8,    53,   626,   626,     6,     3,    27,    52,
       3,    53,     1,    74,    75,     4,     3,     6,     7,     8,
     789,    83,     3,   792,     3,    33,   795,     3,    89,    90,
     807,   800,     6,    94,     3,     3,   658,   659,     3,    51,
       3,   102,   653,   104,   105,   106,   668,   668,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,     9,
       3,   104,   105,     3,     9,     3,    30,     9,     9,     1,
       3,   680,     4,     5,     6,     7,     8,     6,     3,   701,
     701,    52,    70,    53,   695,     3,    53,     3,     3,    69,
       3,     3,    52,     3,    26,    27,   707,     3,     3,   721,
     721,     3,   724,   724,   726,    37,    83,   729,   729,   731,
     731,     3,    51,     6,    51,     6,    51,     4,     3,   741,
      52,     9,    54,     9,   746,   746,     9,   749,   749,    51,
     752,   752,    83,     3,    94,    95,    96,    97,    98,    99,
     100,   101,    74,    75,   104,   105,     6,     3,   759,   771,
       3,   773,     3,   775,    51,   777,    45,    89,    90,     3,
       3,    51,    94,     3,     3,     3,     3,    51,   659,    51,
     102,     6,   104,   105,   106,    51,    51,     3,     3,     3,
       3,     3,   701,   369,   665,   679,   674,   809,   810,   811,
     501,   813,     0,     1,   558,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,   450,    15,   643,   804,
      18,   555,   555,   555,    22,    23,    52,    25,    26,    27,
      28,    29,   456,    31,    32,    32,    34,    35,    36,    37,
     405,    39,    40,    41,    42,    43,    44,    -1,    46,   244,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    74,    75,   104,   105,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    19,    20,    21,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    -1,    -1,    -1,    -1,    70,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    45,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    45,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    45,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    30,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    52,
      -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    37,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,    52,    -1,    54,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    52,
      -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    37,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,    52,    -1,    54,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    52,
      -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    37,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,    52,    -1,    54,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    52,
      -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    37,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,    52,    -1,    54,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    37,    -1,    94,     1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    -1,    -1,    -1,    89,    90,    52,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,     3,     4,     5,     6,     7,     8,     3,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    52,    -1,    54,    -1,    -1,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,     3,    -1,    -1,    89,    90,
      45,    -1,    -1,    94,    89,    90,    -1,    52,    -1,    94,
      -1,   102,    -1,   104,   105,   106,    -1,   102,    -1,   104,
     105,   106,    -1,    68,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,     3,    66,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    45,
      -1,    -1,    -1,    -1,    -1,    -1,    52,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    45,    -1,   104,   105,
      -1,    -1,    -1,    52,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    45,    -1,   104,   105,    -1,    -1,    -1,
      52,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    -1,    -1,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      45,    -1,   104,   105,    -1,    -1,    -1,    52,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    52,    -1,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,     3,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    52,    -1,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,     3,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,     3,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    52,    -1,   104,   105,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    52,     3,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,     3,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    52,    -1,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,     3,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    52,    -1,   104,   105,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
       3,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
       3,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,     3,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    52,
      -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,     3,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,     3,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    52,    -1,   104,   105,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    52,     3,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,     3,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,     3,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    -1,     3,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      52,    -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,     3,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    52,    -1,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,     3,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    52,    -1,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,    -1,
      -1,    26,    27,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    37,     4,     5,     6,     7,     8,    67,    -1,
      45,    -1,    -1,    -1,    -1,    74,    75,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    37,    -1,    -1,    74,
      75,    -1,    -1,   102,    45,   104,   105,   106,    -1,    -1,
      -1,    52,    -1,    54,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    -1,    74,    75,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    98,    26,    27,
      -1,   102,    -1,   104,   105,   106,    -1,    -1,    -1,    37,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    52,    53,    54,    -1,    -1,    -1,
      -1,    -1,    26,    27,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    46,    -1,    -1,    -1,    26,    27,    52,    -1,
      54,    89,    90,    -1,    -1,    -1,    94,    37,     4,     5,
       6,     7,     8,    -1,   102,    -1,   104,   105,   106,    -1,
      74,    75,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    37,    -1,    -1,    74,    75,    -1,    -1,   102,    -1,
     104,   105,   106,    -1,    -1,    51,    52,    -1,    54,    89,
      90,    -1,    -1,    -1,    94,    -1,     4,     5,     6,     7,
       8,    -1,   102,    -1,   104,   105,   106,    -1,    74,    75,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    37,
       4,     5,     6,     7,     8,    -1,   102,    -1,   104,   105,
     106,    -1,    -1,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    52,    -1,
      54,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,    -1,
      74,    75,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,
      -1,    -1,    -1,    -1,    52,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    67,
     104,   105,   106,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    45,    -1,   104,   105,    -1,    -1,
      -1,    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    45,    -1,   104,   105,    -1,    -1,    -1,    52,    53,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,    53,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    69,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
      53,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,    53,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    52,    53,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    67,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,    -1,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    -1,    -1,    -1,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    -1,    -1,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   108,   109,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    46,    48,    49,
      50,    52,    54,    74,    75,    78,    79,    89,    90,    94,
     102,   104,   105,   106,   110,   111,   112,   113,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   132,   133,   134,   136,   137,   145,
     146,   147,   149,   150,   151,   155,   156,   157,   164,   166,
     179,   181,   190,   191,   193,   200,   201,   202,   204,   206,
     214,   215,   216,   217,   219,   220,   221,   224,   242,   247,
     251,   252,   253,   254,   255,   256,   257,   258,   266,   269,
     270,   272,     3,     3,     1,   114,   253,     1,   255,   256,
       1,     3,     1,     3,    14,     1,   256,     1,   253,   255,
     276,     1,   256,     1,     1,   256,     1,   256,   274,     1,
       3,    45,     1,   256,     1,     6,     1,     6,     1,     3,
     256,   248,   267,     1,     6,     7,     1,   256,   258,     1,
       6,     1,     3,    45,     1,   256,     1,     3,     6,   218,
       1,     6,     1,   256,    45,     3,   260,   261,    45,    53,
      67,   256,   275,   277,   256,   255,     1,     3,   274,     3,
     274,   256,   256,   256,   256,   256,   256,   256,   131,    32,
      34,    46,   113,   135,   113,   148,   113,   165,   180,   192,
      47,   209,   212,   213,   113,     1,    52,     1,     6,   222,
     223,   222,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    70,    83,   257,     3,    52,    66,
      71,    72,    73,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   104,   105,    54,
      55,   257,     3,     3,    70,    83,     3,    45,     3,    45,
       3,     3,     3,     3,    45,     3,    45,     3,    45,    83,
      70,    86,     3,     3,     3,     3,     3,     3,     1,    69,
      70,     3,   113,     3,     3,     3,   225,     3,   243,     3,
       3,     6,   249,   250,     1,     6,   207,   208,   268,     3,
       3,     3,     3,     3,     3,    83,     3,    45,     3,     3,
      86,     3,     3,    70,     3,     3,     3,   256,     3,    53,
     256,    53,    45,    67,     1,    70,   260,     1,    70,   260,
      82,    83,     3,     3,     3,   144,   144,   144,   167,   182,
     144,     1,     3,    45,   144,   210,   211,     3,   260,     3,
       3,    70,     9,   222,     3,    98,   256,     6,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   253,   276,
      46,   106,   256,   262,   274,   260,   256,     1,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,     6,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,    98,   256,     6,   253,   256,
     256,   253,     1,   256,     3,   256,   256,     1,    52,   226,
     227,     1,    33,   229,   244,     3,    70,     3,    70,    67,
       1,   252,     1,   256,     6,   260,    53,    53,   256,   256,
     271,   260,    53,   273,   260,    53,   256,   256,     9,   113,
      16,    17,   138,   140,   141,   143,     9,     1,     3,    23,
      24,   168,   173,   175,     1,     3,    23,   173,   183,    30,
     194,   195,   196,   197,     3,    45,     9,   144,   113,     1,
     207,     6,     3,     3,   256,    53,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,    83,   263,   106,     3,
       3,     3,    51,   256,   275,     3,     1,    45,   256,    53,
      83,    69,     3,     3,    45,     3,     3,   260,   235,   229,
       3,     6,   230,   231,     3,   245,     1,   250,   260,   256,
       3,     3,     3,     3,    51,    53,   260,   256,   260,   256,
       3,     1,     3,     1,   256,     9,   139,   142,     3,     3,
       1,     4,     6,     7,     8,   177,   178,     1,     9,   174,
       3,     1,     4,     6,   188,   189,     9,     1,     3,     4,
       6,    86,   198,   199,     9,   196,   144,     3,     9,   205,
       1,   260,    53,   256,   274,     1,    52,   264,     3,     1,
     260,     1,   256,    53,   256,   256,   152,   113,   207,     3,
       6,    38,    46,    47,    88,   201,   236,   237,   239,   240,
       3,    52,   232,    70,     3,   201,   237,   239,   240,   246,
     208,    53,    53,    67,     3,     3,     3,     3,   144,   144,
       3,    45,    69,     3,    45,    70,     3,     3,    45,   176,
       3,    45,     3,    45,    70,     3,     3,   253,     3,    70,
      86,     3,     3,   260,   203,    51,     3,     3,     3,   260,
     209,   259,    51,     3,    45,    68,    19,    20,    21,   113,
     153,   154,   158,   160,   162,     1,   260,    83,     3,     6,
       1,     6,    74,   241,     9,   260,   231,     9,   256,   138,
     171,   172,     4,   169,   170,   178,   144,   113,   186,   187,
     184,   185,   189,     3,   199,   253,    51,   260,     3,    45,
     207,   144,   260,   256,     1,     3,    45,     1,     3,    45,
       1,     3,    45,     9,   153,   228,    51,   256,   238,    83,
       3,     6,     3,    70,     3,     6,    27,   233,   234,   252,
       3,   144,   113,   144,   113,   144,   113,   144,   113,     3,
      45,    51,     1,   260,     9,    51,     3,    45,     3,   159,
     113,     3,   161,   113,     3,   163,   113,     3,   260,     3,
     209,   256,     6,    74,    70,   260,     3,   265,    51,   144,
     144,   144,    51,   144,     3,     6,   234,    51,   260,     3,
       9,     9,     9,     9,    51,     3,     3,     3,     3,     3
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
#line 202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 9:
#line 214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 10:
#line 219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 16:
#line 240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 17:
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 19:
#line 259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 20:
#line 260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 21:
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 25:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true ); COMPILER->defRequired();
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 26:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 27:
#line 283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 49:
#line 309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 50:
#line 311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 51:
#line 316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 52:
#line 320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 53:
#line 324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 54:
#line 328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 55:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 56:
#line 337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 68:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 69:
#line 368 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 417 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 423 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 80:
#line 444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 81:
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 82:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 83:
#line 460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 84:
#line 461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 85:
#line 465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 86:
#line 466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 87:
#line 467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 88:
#line 471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 89:
#line 479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 90:
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 91:
#line 496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 92:
#line 497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 93:
#line 501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 94:
#line 502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 97:
#line 509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 100:
#line 519 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 101:
#line 524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 103:
#line 536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 104:
#line 537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 106:
#line 542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 107:
#line 549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 108:
#line 558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 109:
#line 566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 110:
#line 576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 111:
#line 585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 112:
#line 592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 113:
#line 599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 114:
#line 607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (2)].fal_stat) != 0 )
         {
            Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>((yyvsp[(1) - (2)].fal_stat));
            if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
                f->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
            (yyval.fal_stat) = f;
         }
         else
            delete (yyvsp[(2) - (2)].fal_stat);
      ;}
    break;

  case 115:
#line 622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 116:
#line 626 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 117:
#line 631 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 118:
#line 638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 119:
#line 642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 120:
#line 647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 121:
#line 656 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 122:
#line 672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 123:
#line 680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (6)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (6)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (6)].fal_val) );
         if ( (yyvsp[(6) - (6)].fal_stat) != 0 )
             f->children().push_back( (yyvsp[(6) - (6)].fal_stat) );
      ;}
    break;

  case 124:
#line 694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 127:
#line 703 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 131:
#line 717 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 132:
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 133:
#line 738 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 134:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 135:
#line 748 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 136:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 137:
#line 760 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 138:
#line 769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 139:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 140:
#line 790 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 141:
#line 792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 142:
#line 800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 143:
#line 804 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 144:
#line 816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 145:
#line 817 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 146:
#line 825 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 147:
#line 829 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->allBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->allBlock() );
      ;}
    break;

  case 148:
#line 841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 149:
#line 843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 150:
#line 851 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 151:
#line 855 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 152:
#line 863 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 153:
#line 872 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 154:
#line 874 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 157:
#line 883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 159:
#line 889 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 161:
#line 899 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 162:
#line 907 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 163:
#line 911 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 166:
#line 933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 168:
#line 942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 172:
#line 956 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 174:
#line 960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 177:
#line 972 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 178:
#line 981 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 179:
#line 993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 180:
#line 1004 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 181:
#line 1015 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 182:
#line 1035 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 183:
#line 1043 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 184:
#line 1052 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 185:
#line 1054 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 188:
#line 1063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 190:
#line 1069 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 192:
#line 1079 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 193:
#line 1088 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 194:
#line 1092 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 196:
#line 1104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 197:
#line 1114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 201:
#line 1128 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 202:
#line 1140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 203:
#line 1161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 204:
#line 1165 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 205:
#line 1169 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 206:
#line 1177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 207:
#line 1184 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 208:
#line 1194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 210:
#line 1203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 216:
#line 1223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 217:
#line 1241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 218:
#line 1261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 219:
#line 1271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 220:
#line 1282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 223:
#line 1295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 224:
#line 1307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 225:
#line 1329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 226:
#line 1330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 227:
#line 1342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 228:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 230:
#line 1357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 231:
#line 1358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 232:
#line 1361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 234:
#line 1366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 235:
#line 1367 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 236:
#line 1374 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 240:
#line 1435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 242:
#line 1452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 243:
#line 1458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 244:
#line 1463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 245:
#line 1469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 247:
#line 1478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 249:
#line 1483 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 250:
#line 1493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 251:
#line 1496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 252:
#line 1505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 253:
#line 1512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 254:
#line 1522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 255:
#line 1528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 256:
#line 1540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 257:
#line 1550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1567 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 260:
#line 1576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 263:
#line 1596 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 264:
#line 1610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 265:
#line 1617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 267:
#line 1625 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 269:
#line 1629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 271:
#line 1635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 272:
#line 1639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 275:
#line 1648 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 276:
#line 1659 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 277:
#line 1693 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 279:
#line 1721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 282:
#line 1729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 283:
#line 1730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      ;}
    break;

  case 288:
#line 1747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 289:
#line 1780 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 290:
#line 1785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(3) - (5)].fal_adecl);
   ;}
    break;

  case 291:
#line 1791 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 292:
#line 1792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 294:
#line 1798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 295:
#line 1806 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 299:
#line 1816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 300:
#line 1819 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 302:
#line 1841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 303:
#line 1865 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 304:
#line 1876 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 305:
#line 1898 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 308:
#line 1928 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 309:
#line 1935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 310:
#line 1943 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 311:
#line 1949 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 312:
#line 1955 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 313:
#line 1968 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 314:
#line 2008 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 316:
#line 2033 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 320:
#line 2045 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 321:
#line 2048 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 323:
#line 2076 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 324:
#line 2081 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 327:
#line 2096 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 328:
#line 2103 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 329:
#line 2118 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 330:
#line 2119 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 331:
#line 2120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 332:
#line 2130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 333:
#line 2131 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 334:
#line 2132 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 335:
#line 2133 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 336:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 338:
#line 2156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 339:
#line 2157 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 341:
#line 2169 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 342:
#line 2174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 343:
#line 2179 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 344:
#line 2185 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 347:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 348:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 349:
#line 2198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 351:
#line 2200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 358:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 359:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 360:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 361:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 362:
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 363:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 364:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 365:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 366:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 367:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 377:
#line 2228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 382:
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 383:
#line 2234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 384:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 387:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 388:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 389:
#line 2246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 390:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 395:
#line 2258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(3) - (5)].fal_val); ;}
    break;

  case 396:
#line 2263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 397:
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 398:
#line 2269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 399:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 400:
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (6)].fal_val), new Falcon::Value( (yyvsp[(4) - (6)].fal_adecl) ) ) );
      ;}
    break;

  case 401:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (4)].fal_val), 0 ) );
      ;}
    break;

  case 402:
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 403:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(4) - (8)].fal_adecl);
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 408:
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

  case 409:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 411:
#line 2352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 412:
#line 2353 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 413:
#line 2357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 414:
#line 2365 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 415:
#line 2396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 417:
#line 2410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 418:
#line 2419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 419:
#line 2424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 420:
#line 2431 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 421:
#line 2438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 422:
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 423:
#line 2449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_adecl) );
      ;}
    break;

  case 424:
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 425:
#line 2453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_adecl) );
      ;}
    break;

  case 426:
#line 2460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 427:
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) ); ;}
    break;

  case 428:
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 429:
#line 2463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_ddecl) );
      ;}
    break;

  case 430:
#line 2470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 431:
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 432:
#line 2475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 433:
#line 2476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (4)].fal_adecl)->pushBack( (yyvsp[(4) - (4)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (4)].fal_adecl); ;}
    break;

  case 434:
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 435:
#line 2487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 436:
#line 2494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 437:
#line 2495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (6)].fal_ddecl)->pushBack( (yyvsp[(4) - (6)].fal_val), (yyvsp[(6) - (6)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (6)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6149 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


