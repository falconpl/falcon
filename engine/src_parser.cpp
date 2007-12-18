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
#define YYLAST   6068

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  172
/* YYNRULES -- Number of rules.  */
#define YYNRULES  439
/* YYNRULES -- Number of states.  */
#define YYNSTATES  833

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
     399,   403,   404,   414,   415,   423,   429,   430,   433,   435,
     437,   439,   441,   445,   449,   453,   456,   460,   463,   467,
     471,   473,   474,   481,   485,   489,   490,   497,   501,   505,
     506,   513,   517,   521,   522,   529,   533,   537,   538,   541,
     545,   547,   548,   554,   555,   561,   562,   568,   569,   575,
     576,   577,   581,   582,   584,   587,   590,   593,   595,   599,
     601,   603,   605,   609,   611,   612,   619,   623,   627,   628,
     631,   635,   637,   638,   644,   645,   651,   652,   658,   659,
     665,   667,   671,   672,   674,   676,   682,   687,   691,   695,
     696,   703,   706,   710,   711,   713,   715,   718,   721,   724,
     729,   733,   739,   743,   745,   749,   751,   753,   757,   761,
     767,   770,   778,   779,   789,   793,   801,   802,   811,   814,
     815,   817,   822,   824,   825,   826,   832,   833,   837,   840,
     844,   847,   851,   855,   859,   863,   869,   875,   879,   885,
     891,   895,   898,   902,   906,   908,   912,   917,   921,   924,
     928,   931,   935,   936,   938,   942,   945,   949,   952,   953,
     962,   966,   969,   970,   976,   977,   985,   986,   989,   991,
     995,   998,   999,  1005,  1007,  1011,  1013,  1015,  1017,  1018,
    1021,  1023,  1025,  1027,  1029,  1030,  1038,  1044,  1049,  1050,
    1054,  1058,  1060,  1063,  1067,  1072,  1073,  1082,  1085,  1088,
    1089,  1092,  1094,  1096,  1098,  1100,  1101,  1106,  1108,  1112,
    1116,  1118,  1121,  1125,  1129,  1131,  1133,  1135,  1137,  1139,
    1141,  1143,  1145,  1147,  1150,  1155,  1161,  1165,  1167,  1169,
    1173,  1176,  1180,  1184,  1188,  1192,  1196,  1200,  1204,  1208,
    1212,  1216,  1219,  1224,  1229,  1233,  1236,  1239,  1242,  1245,
    1249,  1253,  1257,  1261,  1265,  1269,  1273,  1277,  1281,  1284,
    1288,  1292,  1296,  1300,  1304,  1307,  1310,  1313,  1315,  1317,
    1321,  1326,  1332,  1335,  1337,  1339,  1341,  1343,  1349,  1353,
    1358,  1363,  1369,  1376,  1381,  1382,  1391,  1392,  1394,  1396,
    1399,  1400,  1407,  1414,  1415,  1424,  1427,  1428,  1434,  1436,
    1439,  1445,  1451,  1456,  1460,  1463,  1468,  1469,  1476,  1480,
    1485,  1486,  1493,  1495,  1499,  1501,  1506,  1508,  1512,  1516
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     108,     0,    -1,   109,    -1,    -1,   109,   110,    -1,   111,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   112,    -1,
     202,    -1,   225,    -1,   243,    -1,   113,    -1,   217,    -1,
     218,    -1,   220,    -1,    39,     6,     3,    -1,    39,     7,
       3,    -1,    39,     1,     3,    -1,   115,    -1,     3,    -1,
      46,     1,     3,    -1,    34,     1,     3,    -1,    32,     1,
       3,    -1,     1,     3,    -1,   254,    83,   257,    -1,   114,
      70,   254,    83,   257,    -1,   257,     3,    -1,   116,    -1,
     117,    -1,   118,    -1,   130,    -1,   147,    -1,   151,    -1,
     165,    -1,   180,    -1,   134,    -1,   145,    -1,   146,    -1,
     191,    -1,   192,    -1,   201,    -1,   252,    -1,   248,    -1,
     215,    -1,   216,    -1,   156,    -1,   157,    -1,   158,    -1,
      10,   114,     3,    -1,    10,     1,     3,    -1,   256,    83,
     257,     3,    -1,   256,    83,   106,   106,     3,    -1,   256,
      83,   275,     3,    -1,   256,    83,   263,     3,    -1,   256,
      70,   277,    83,   257,     3,    -1,   256,    70,   277,    83,
     275,     3,    -1,   119,    -1,   120,    -1,   121,    -1,   122,
      -1,   123,    -1,   124,    -1,   125,    -1,   126,    -1,   127,
      -1,   128,    -1,   129,    -1,   257,    66,   257,     3,    -1,
     256,    65,   257,     3,    -1,   256,    64,   257,     3,    -1,
     256,    63,   257,     3,    -1,   256,    62,   257,     3,    -1,
     256,    56,   257,     3,    -1,   256,    61,   257,     3,    -1,
     256,    60,   257,     3,    -1,   256,    59,   257,     3,    -1,
     256,    57,   257,     3,    -1,   256,    58,   257,     3,    -1,
      -1,   132,   131,   144,     9,     3,    -1,   133,   113,    -1,
      11,   257,     3,    -1,    49,    -1,    11,     1,     3,    -1,
      11,   257,    45,    -1,    49,    45,    -1,    11,     1,    45,
      -1,    -1,   136,   135,   144,   138,     9,     3,    -1,   137,
     113,    -1,    15,   257,     3,    -1,    15,     1,     3,    -1,
      15,   257,    45,    -1,    15,     1,    45,    -1,    -1,   141,
      -1,    -1,   140,   139,   144,    -1,    16,     3,    -1,    16,
       1,     3,    -1,    -1,   143,   142,   144,   138,    -1,    17,
     257,     3,    -1,    17,     1,     3,    -1,    -1,   144,   113,
      -1,    12,     3,    -1,    12,     1,     3,    -1,    13,     3,
      -1,    13,    14,     3,    -1,    13,     1,     3,    -1,    -1,
     149,   148,   144,     9,     3,    -1,   150,   113,    -1,    18,
     256,    83,   257,    69,   257,     3,    -1,    18,   256,    83,
     257,    69,   257,    68,   257,     3,    -1,    18,     1,     3,
      -1,    18,   256,    83,   257,    69,   257,    45,    -1,    18,
     256,    83,   257,    69,   257,    68,   257,    45,    -1,    18,
       1,    45,    -1,    -1,    18,   277,    86,   257,     3,   152,
     154,     9,     3,    -1,    -1,    18,   277,    86,   257,    45,
     153,   113,    -1,    18,   277,    86,     1,     3,    -1,    -1,
     155,   154,    -1,   113,    -1,   159,    -1,   161,    -1,   163,
      -1,    48,   257,     3,    -1,    48,     1,     3,    -1,   100,
     275,     3,    -1,   100,     3,    -1,    79,   275,     3,    -1,
      79,     3,    -1,   100,     1,     3,    -1,    79,     1,     3,
      -1,    50,    -1,    -1,    19,     3,   160,   144,     9,     3,
      -1,    19,    45,   113,    -1,    19,     1,     3,    -1,    -1,
      20,     3,   162,   144,     9,     3,    -1,    20,    45,   113,
      -1,    20,     1,     3,    -1,    -1,    21,     3,   164,   144,
       9,     3,    -1,    21,    45,   113,    -1,    21,     1,     3,
      -1,    -1,   167,   166,   168,   174,     9,     3,    -1,    22,
     257,     3,    -1,    22,     1,     3,    -1,    -1,   168,   169,
      -1,   168,     1,     3,    -1,     3,    -1,    -1,    23,   178,
       3,   170,   144,    -1,    -1,    23,   178,    45,   171,   113,
      -1,    -1,    23,     1,     3,   172,   144,    -1,    -1,    23,
       1,    45,   173,   113,    -1,    -1,    -1,   176,   175,   177,
      -1,    -1,    24,    -1,    24,     1,    -1,     3,   144,    -1,
      45,   113,    -1,   179,    -1,   178,    70,   179,    -1,     8,
      -1,     4,    -1,     7,    -1,     4,    69,     4,    -1,     6,
      -1,    -1,   182,   181,   183,   174,     9,     3,    -1,    25,
     257,     3,    -1,    25,     1,     3,    -1,    -1,   183,   184,
      -1,   183,     1,     3,    -1,     3,    -1,    -1,    23,   189,
       3,   185,   144,    -1,    -1,    23,   189,    45,   186,   113,
      -1,    -1,    23,     1,     3,   187,   144,    -1,    -1,    23,
       1,    45,   188,   113,    -1,   190,    -1,   189,    70,   190,
      -1,    -1,     4,    -1,     6,    -1,    28,   275,    69,   257,
       3,    -1,    28,   275,     1,     3,    -1,    28,     1,     3,
      -1,    29,    45,   113,    -1,    -1,   194,   193,   144,   195,
       9,     3,    -1,    29,     3,    -1,    29,     1,     3,    -1,
      -1,   196,    -1,   197,    -1,   196,   197,    -1,   198,   144,
      -1,    30,     3,    -1,    30,    86,   254,     3,    -1,    30,
     199,     3,    -1,    30,   199,    86,   254,     3,    -1,    30,
       1,     3,    -1,   200,    -1,   199,    70,   200,    -1,     4,
      -1,     6,    -1,    31,   257,     3,    -1,    31,     1,     3,
      -1,   203,   210,   144,     9,     3,    -1,   205,   113,    -1,
     207,    52,   261,   208,   261,    51,     3,    -1,    -1,   207,
      52,   261,   208,     1,   204,   261,    51,     3,    -1,   207,
       1,     3,    -1,   207,    52,   261,   208,   261,    51,    45,
      -1,    -1,   207,    52,   261,     1,   206,   261,    51,    45,
      -1,    46,     6,    -1,    -1,   209,    -1,   208,    70,   261,
     209,    -1,     6,    -1,    -1,    -1,   213,   211,   144,     9,
       3,    -1,    -1,   214,   212,   113,    -1,    47,     3,    -1,
      47,     1,     3,    -1,    47,    45,    -1,    47,     1,    45,
      -1,    40,   259,     3,    -1,    40,     1,     3,    -1,    43,
     257,     3,    -1,    43,   257,    86,   257,     3,    -1,    43,
     257,    86,     1,     3,    -1,    43,     1,     3,    -1,    41,
       6,    83,   253,     3,    -1,    41,     6,    83,     1,     3,
      -1,    41,     1,     3,    -1,    44,     3,    -1,    44,   219,
       3,    -1,    44,     1,     3,    -1,     6,    -1,   219,    70,
       6,    -1,   221,   224,     9,     3,    -1,   222,   223,     3,
      -1,    42,     3,    -1,    42,     1,     3,    -1,    42,    45,
      -1,    42,     1,    45,    -1,    -1,     6,    -1,   223,    70,
       6,    -1,   223,     3,    -1,   224,   223,     3,    -1,     1,
       3,    -1,    -1,    32,     6,   226,   227,   236,   241,     9,
       3,    -1,   228,   230,     3,    -1,     1,     3,    -1,    -1,
      52,   261,   208,   261,    51,    -1,    -1,    52,   261,   208,
       1,   229,   261,    51,    -1,    -1,    33,   231,    -1,   232,
      -1,   231,    70,   232,    -1,     6,   233,    -1,    -1,    52,
     261,   234,   261,    51,    -1,   235,    -1,   234,    70,   235,
      -1,   253,    -1,     6,    -1,    27,    -1,    -1,   236,   237,
      -1,     3,    -1,   202,    -1,   240,    -1,   238,    -1,    -1,
      38,     3,   239,   210,   144,     9,     3,    -1,    47,     6,
      83,   257,     3,    -1,     6,    83,   257,     3,    -1,    -1,
      88,   242,     3,    -1,    88,     1,     3,    -1,     6,    -1,
      74,     6,    -1,   242,    70,     6,    -1,   242,    70,    74,
       6,    -1,    -1,    34,     6,   244,   245,   246,   241,     9,
       3,    -1,   230,     3,    -1,     1,     3,    -1,    -1,   246,
     247,    -1,     3,    -1,   202,    -1,   240,    -1,   238,    -1,
      -1,    36,   249,   250,     3,    -1,   251,    -1,   250,    70,
     251,    -1,   250,    70,     1,    -1,     6,    -1,    35,     3,
      -1,    35,   257,     3,    -1,    35,     1,     3,    -1,     8,
      -1,     4,    -1,     5,    -1,     7,    -1,     6,    -1,   254,
      -1,    27,    -1,    26,    -1,   255,    -1,   256,   258,    -1,
     256,    54,   257,    53,    -1,   256,    54,    98,   257,    53,
      -1,   256,    55,     6,    -1,   253,    -1,   256,    -1,   257,
      95,   257,    -1,    94,   257,    -1,   257,    94,   257,    -1,
     257,    98,   257,    -1,   257,    97,   257,    -1,   257,    96,
     257,    -1,   257,    99,   257,    -1,   257,    93,   257,    -1,
     257,    92,   257,    -1,   257,    91,   257,    -1,   257,   101,
     257,    -1,   257,   100,   257,    -1,   102,   257,    -1,    75,
     256,    82,   257,    -1,    75,   256,    83,   257,    -1,   257,
      80,   257,    -1,   257,   105,    -1,   105,   257,    -1,   257,
     104,    -1,   104,   257,    -1,   257,    81,   257,    -1,   257,
      82,   257,    -1,   257,    83,   257,    -1,   257,    79,   257,
      -1,   257,    78,   257,    -1,   257,    77,   257,    -1,   257,
      76,   257,    -1,   257,    73,   257,    -1,   257,    72,   257,
      -1,    74,   257,    -1,   257,    88,   257,    -1,   257,    87,
     257,    -1,   257,    86,   257,    -1,   257,    85,   257,    -1,
     257,    84,     6,    -1,   106,   257,    -1,    90,   257,    -1,
      89,   257,    -1,   267,    -1,   259,    -1,   259,    55,     6,
      -1,   259,    54,   257,    53,    -1,   259,    54,    98,   257,
      53,    -1,   259,   258,    -1,   270,    -1,   271,    -1,   273,
      -1,   258,    -1,    52,   261,   257,   261,    51,    -1,    54,
      45,    53,    -1,    54,   257,    45,    53,    -1,    54,    45,
     257,    53,    -1,    54,   257,    45,   257,    53,    -1,   257,
      52,   261,   276,   261,    51,    -1,   257,    52,   261,    51,
      -1,    -1,   257,    52,   261,   276,     1,   260,   261,    51,
      -1,    -1,   262,    -1,     3,    -1,   262,     3,    -1,    -1,
      46,   264,   265,   210,   144,     9,    -1,    52,   261,   208,
     261,    51,     3,    -1,    -1,    52,   261,   208,     1,   266,
     261,    51,     3,    -1,     1,     3,    -1,    -1,    37,   268,
     269,    67,   257,    -1,   208,    -1,     1,     3,    -1,   257,
      71,   257,    45,   257,    -1,   257,    71,   257,    45,     1,
      -1,   257,    71,   257,     1,    -1,   257,    71,     1,    -1,
      54,    53,    -1,    54,   276,   261,    53,    -1,    -1,    54,
     276,     1,   272,   261,    53,    -1,    54,    67,    53,    -1,
      54,   278,   261,    53,    -1,    -1,    54,   278,     1,   274,
     261,    53,    -1,   257,    -1,   275,    70,   257,    -1,   257,
      -1,   276,    70,   261,   257,    -1,   254,    -1,   277,    70,
     254,    -1,   257,    67,   257,    -1,   278,    70,   261,   257,
      67,   257,    -1
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
     646,   656,   655,   680,   679,   705,   709,   711,   715,   722,
     723,   724,   728,   741,   749,   753,   759,   765,   772,   777,
     786,   796,   796,   810,   818,   822,   822,   835,   843,   847,
     847,   861,   869,   873,   873,   890,   891,   898,   900,   901,
     905,   907,   906,   917,   917,   929,   929,   941,   941,   957,
     960,   959,   972,   973,   974,   977,   978,   984,   985,   989,
     998,  1010,  1021,  1032,  1053,  1053,  1070,  1071,  1078,  1080,
    1081,  1085,  1087,  1086,  1097,  1097,  1110,  1110,  1122,  1122,
    1140,  1141,  1144,  1145,  1157,  1178,  1182,  1187,  1195,  1202,
    1201,  1220,  1221,  1224,  1226,  1230,  1231,  1235,  1240,  1258,
    1278,  1288,  1299,  1307,  1308,  1312,  1324,  1347,  1348,  1355,
    1365,  1374,  1375,  1375,  1379,  1383,  1384,  1384,  1391,  1445,
    1447,  1448,  1452,  1467,  1470,  1469,  1481,  1480,  1495,  1496,
    1500,  1501,  1510,  1514,  1522,  1529,  1539,  1545,  1557,  1567,
    1572,  1584,  1593,  1600,  1608,  1613,  1625,  1632,  1642,  1643,
    1646,  1647,  1650,  1652,  1656,  1663,  1664,  1665,  1677,  1676,
    1735,  1738,  1744,  1746,  1747,  1747,  1753,  1755,  1759,  1760,
    1764,  1798,  1800,  1809,  1810,  1814,  1815,  1824,  1827,  1829,
    1833,  1834,  1837,  1855,  1859,  1859,  1893,  1915,  1942,  1944,
    1945,  1952,  1960,  1966,  1972,  1986,  1985,  2049,  2050,  2056,
    2058,  2062,  2063,  2066,  2085,  2094,  2093,  2111,  2112,  2113,
    2120,  2136,  2137,  2138,  2148,  2149,  2150,  2151,  2155,  2173,
    2174,  2175,  2186,  2187,  2192,  2197,  2203,  2212,  2213,  2214,
    2215,  2216,  2217,  2218,  2219,  2220,  2221,  2222,  2223,  2224,
    2225,  2226,  2227,  2229,  2231,  2232,  2233,  2234,  2235,  2236,
    2237,  2238,  2239,  2240,  2241,  2242,  2243,  2244,  2245,  2246,
    2247,  2248,  2249,  2250,  2251,  2252,  2253,  2254,  2255,  2256,
    2260,  2264,  2268,  2272,  2273,  2274,  2275,  2276,  2281,  2284,
    2287,  2290,  2296,  2302,  2307,  2307,  2315,  2317,  2321,  2322,
    2327,  2326,  2369,  2370,  2370,  2374,  2383,  2382,  2426,  2427,
    2436,  2441,  2448,  2455,  2465,  2466,  2470,  2470,  2478,  2479,
    2480,  2480,  2488,  2489,  2493,  2494,  2498,  2505,  2512,  2513
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
  "@6", "@7", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "@8", "last_loop_block", "@9", "all_loop_block",
  "@10", "switch_statement", "@11", "switch_decl", "case_list",
  "case_statement", "@12", "@13", "@14", "@15", "default_statement", "@16",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "@17", "select_decl", "selcase_list",
  "selcase_statement", "@18", "@19", "@20", "@21",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@22", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@23", "func_decl_short", "@24", "func_begin", "param_list",
  "param_symbol", "static_block", "@25", "@26", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@27",
  "class_def_inner", "class_param_list", "@28", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@29", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@30", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@31", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@32",
  "opt_eol", "eol_seq", "nameless_func", "@33", "nameless_func_decl_inner",
  "@34", "lambda_expr", "@35", "lambda_expr_inner", "iif_expr",
  "array_decl", "@36", "dict_decl", "@37", "expression_list",
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
     150,   152,   151,   153,   151,   151,   154,   154,   155,   155,
     155,   155,   156,   156,   157,   157,   157,   157,   157,   157,
     158,   160,   159,   159,   159,   162,   161,   161,   161,   164,
     163,   163,   163,   166,   165,   167,   167,   168,   168,   168,
     169,   170,   169,   171,   169,   172,   169,   173,   169,   174,
     175,   174,   176,   176,   176,   177,   177,   178,   178,   179,
     179,   179,   179,   179,   181,   180,   182,   182,   183,   183,
     183,   184,   185,   184,   186,   184,   187,   184,   188,   184,
     189,   189,   190,   190,   190,   191,   191,   191,   192,   193,
     192,   194,   194,   195,   195,   196,   196,   197,   198,   198,
     198,   198,   198,   199,   199,   200,   200,   201,   201,   202,
     202,   203,   204,   203,   203,   205,   206,   205,   207,   208,
     208,   208,   209,   210,   211,   210,   212,   210,   213,   213,
     214,   214,   215,   215,   216,   216,   216,   216,   217,   217,
     217,   218,   218,   218,   219,   219,   220,   220,   221,   221,
     222,   222,   223,   223,   223,   224,   224,   224,   226,   225,
     227,   227,   228,   228,   229,   228,   230,   230,   231,   231,
     232,   233,   233,   234,   234,   235,   235,   235,   236,   236,
     237,   237,   237,   237,   239,   238,   240,   240,   241,   241,
     241,   242,   242,   242,   242,   244,   243,   245,   245,   246,
     246,   247,   247,   247,   247,   249,   248,   250,   250,   250,
     251,   252,   252,   252,   253,   253,   253,   253,   254,   255,
     255,   255,   256,   256,   256,   256,   256,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   258,   258,
     258,   258,   259,   259,   260,   259,   261,   261,   262,   262,
     264,   263,   265,   266,   265,   265,   268,   267,   269,   269,
     270,   270,   270,   270,   271,   271,   272,   271,   273,   273,
     274,   273,   275,   275,   276,   276,   277,   277,   278,   278
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
       3,     0,     9,     0,     7,     5,     0,     2,     1,     1,
       1,     1,     3,     3,     3,     2,     3,     2,     3,     3,
       1,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     0,
       0,     3,     0,     1,     2,     2,     2,     1,     3,     1,
       1,     1,     3,     1,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       1,     3,     0,     1,     1,     5,     4,     3,     3,     0,
       6,     2,     3,     0,     1,     1,     2,     2,     2,     4,
       3,     5,     3,     1,     3,     1,     1,     3,     3,     5,
       2,     7,     0,     9,     3,     7,     0,     8,     2,     0,
       1,     4,     1,     0,     0,     5,     0,     3,     2,     3,
       2,     3,     3,     3,     3,     5,     5,     3,     5,     5,
       3,     2,     3,     3,     1,     3,     4,     3,     2,     3,
       2,     3,     0,     1,     3,     2,     3,     2,     0,     8,
       3,     2,     0,     5,     0,     7,     0,     2,     1,     3,
       2,     0,     5,     1,     3,     1,     1,     1,     0,     2,
       1,     1,     1,     1,     0,     7,     5,     4,     0,     3,
       3,     1,     2,     3,     4,     0,     8,     2,     2,     0,
       2,     1,     1,     1,     1,     0,     4,     1,     3,     3,
       1,     2,     3,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     4,     5,     3,     1,     1,     3,
       2,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     2,     4,     4,     3,     2,     2,     2,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       3,     3,     3,     3,     2,     2,     2,     1,     1,     3,
       4,     5,     2,     1,     1,     1,     1,     5,     3,     4,
       4,     5,     6,     4,     0,     8,     0,     1,     1,     2,
       0,     6,     6,     0,     8,     2,     0,     5,     1,     2,
       5,     5,     4,     3,     2,     4,     0,     6,     3,     4,
       0,     6,     1,     3,     1,     4,     1,     3,     3,     6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    20,   335,   336,   338,   337,
     334,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   341,   340,     0,     0,     0,     0,     0,     0,   325,
     416,     0,     0,     0,     0,     0,     0,     0,     0,    83,
     140,   406,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    12,    19,    28,
      29,    30,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    31,    79,     0,    36,    88,     0,    37,
      38,    32,   112,     0,    33,    46,    47,    48,    34,   153,
      35,   184,    39,    40,   209,    41,     9,   243,     0,     0,
      44,    45,    13,    14,    15,     0,   272,    10,    11,    43,
      42,   347,   339,   342,   348,     0,   396,   388,   387,   393,
     394,   395,    24,     6,     0,     0,     0,     0,   348,     0,
       0,   107,     0,   109,     0,     0,     0,     0,   339,     0,
       0,     0,     0,     0,     0,     0,     0,   432,     0,     0,
     211,     0,     0,     0,     0,   278,     0,   315,     0,   331,
       0,     0,     0,     0,     0,     0,     0,     0,   388,     0,
       0,     0,   268,   270,     0,     0,     0,   261,   264,     0,
       0,   238,     0,     0,    86,   408,     0,   407,     0,   424,
       0,   434,     0,     0,   378,     0,     0,   137,     0,   386,
     385,   350,     0,   135,     0,   361,   368,   366,   384,   105,
       0,     0,     0,    81,   105,    90,   105,   114,   157,   188,
     105,     0,   105,   244,   246,   230,     0,   406,     0,   273,
       0,   272,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   343,    27,   406,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   367,   365,
       0,     0,   392,    50,    49,     0,     0,    84,    87,    82,
      85,   108,   111,   110,    92,    94,    91,    93,   117,   120,
       0,     0,     0,   156,   155,     7,   187,   186,   207,     0,
       0,     0,   212,   208,   228,   227,    23,     0,    22,     0,
     333,   332,   330,     0,   327,     0,   242,   418,   240,     0,
      18,    16,    17,   253,   252,   260,     0,   269,   271,   257,
     254,     0,   263,   262,     0,    21,   133,   132,   406,   409,
     398,     0,   428,     0,     0,   426,   406,     0,   430,   406,
       0,     0,     0,   139,   136,   138,   134,     0,     0,     0,
       0,     0,     0,     0,   248,   250,     0,   105,     0,   234,
       0,   277,   275,     0,     0,     0,   267,     0,     0,   346,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     436,     0,   410,     0,   432,     0,     0,     0,     0,   423,
       0,   377,   376,   375,   374,   373,   372,   364,   369,   370,
     371,   383,   382,   381,   380,   379,   358,   357,   356,   351,
     349,   354,   353,   352,   355,   360,   359,     0,     0,   389,
       0,    25,     0,   437,     0,     0,   206,     0,   433,     0,
     406,   298,   286,     0,     0,     0,   319,   326,     0,   419,
     406,     0,     0,     0,     0,   381,   265,     0,   400,   399,
       0,   438,   406,     0,   425,   406,     0,   429,   362,   363,
       0,   106,     0,     0,     0,    97,    96,   101,     0,     0,
     160,     0,     0,   158,     0,   170,     0,   191,     0,     0,
     189,     0,     0,   214,   215,   105,   249,   251,     0,     0,
     247,   236,     0,   274,   266,   276,     0,   344,    73,    77,
      78,    76,    75,    74,    72,    71,    70,    69,     0,     0,
       0,    51,    54,    53,   403,   434,     0,    68,   422,     0,
       0,   390,     0,     0,   125,   121,   123,   205,   281,   239,
     308,     0,   318,   291,   287,   288,   317,   308,   329,   328,
       0,   417,   259,   258,   256,   255,   397,   401,     0,   435,
       0,     0,    80,     0,    99,     0,     0,     0,   105,   105,
     113,   159,     0,   180,   183,   181,   179,     0,   177,   174,
       0,     0,   190,     0,   203,   204,     0,   200,     0,     0,
     218,   225,   226,     0,     0,   223,     0,   216,     0,   229,
       0,   406,   232,     0,   345,   432,     0,     0,   406,   243,
      52,   404,     0,   421,   420,   391,    26,     0,     0,     0,
       0,   300,     0,     0,     0,     0,     0,   301,   299,   303,
     302,     0,   280,   406,   290,     0,   321,   322,   324,   323,
       0,   320,   241,   427,   431,     0,   100,   104,   103,    89,
       0,     0,   165,   167,     0,   161,   163,     0,   154,   105,
       0,   171,   196,   198,   192,   194,   202,   185,   222,     0,
     220,     0,     0,   210,   245,     0,   406,     0,    55,    56,
     415,   239,   105,   406,   402,   115,   118,     0,     0,     0,
       0,   128,     0,     0,   129,   130,   131,   124,   284,     0,
       0,   304,     0,     0,   311,     0,     0,     0,     0,   289,
       0,   439,   102,   105,     0,   182,   105,     0,   178,     0,
     176,   105,     0,   105,     0,   201,   219,   224,     0,     0,
       0,   231,   235,     0,     0,     0,     0,     0,   141,     0,
       0,   145,     0,     0,   149,     0,     0,   127,   406,   283,
       0,   243,     0,   310,   312,   309,     0,   279,   296,   297,
     406,   293,   295,   316,     0,   168,     0,   164,     0,   199,
       0,   195,   221,   237,     0,   413,     0,   411,   405,   116,
     119,   144,   105,   143,   148,   105,   147,   152,   105,   151,
     122,     0,   307,   105,     0,   313,     0,     0,     0,   233,
     406,     0,     0,     0,     0,   285,     0,   306,   314,   294,
     292,     0,   412,     0,     0,     0,     0,     0,   142,   146,
     150,   305,   414
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    54,    55,    56,   481,   125,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,   209,    74,    75,    76,   214,    77,
      78,   484,   578,   485,   486,   579,   487,   367,    79,    80,
      81,   216,    82,    83,    84,   628,   629,   702,   703,    85,
      86,    87,   704,   792,   705,   795,   706,   798,    88,   218,
      89,   370,   493,   726,   727,   723,   724,   494,   591,   495,
     671,   587,   588,    90,   219,    91,   371,   500,   733,   734,
     731,   732,   596,   597,    92,    93,   220,    94,   502,   503,
     504,   505,   604,   605,    95,    96,    97,   686,    98,   611,
      99,   327,   328,   222,   377,   378,   223,   224,   100,   101,
     102,   103,   179,   104,   105,   106,   230,   231,   107,   317,
     451,   452,   758,   455,   554,   555,   644,   770,   771,   550,
     638,   639,   761,   640,   641,   716,   108,   319,   456,   557,
     651,   109,   161,   323,   324,   110,   111,   112,   113,   128,
     115,   116,   117,   693,   186,   187,   405,   529,   619,   810,
     118,   162,   329,   119,   120,   472,   121,   475,   148,   192,
     140,   193
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -615
static const yytype_int16 yypact[] =
{
    -615,     9,   815,  -615,    25,  -615,  -615,  -615,  -615,  -615,
    -615,    34,    39,   519,   297,   296,   681,   455,  2990,    40,
    3045,  -615,  -615,  3100,   171,  3155,   218,   453,   103,  -615,
    -615,   418,  3210,   459,   187,  3265,   406,   469,  3320,    23,
    -615,    87,  5115,  5396,   431,   389,  5396,  5396,  5396,   495,
    5396,  5396,  5396,  5396,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  2935,  -615,  -615,  2935,  -615,
    -615,  -615,  -615,  2935,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,    90,  2935,    30,
    -615,  -615,  -615,  -615,  -615,    19,   170,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  1028,  3741,  -615,   451,  -615,  -615,
    -615,  -615,  -615,  -615,   177,    56,   111,    29,   511,  3798,
     217,  -615,   247,  -615,   286,   130,  3855,   276,   100,   344,
     298,   299,  4019,   305,   353,  4056,   357,  5852,    33,   368,
    -615,  2935,   375,  4106,   379,  -615,   388,  -615,   400,  -615,
    4143,   408,    55,   415,   419,   443,   449,  5852,   260,   466,
     390,   290,  -615,  -615,   477,  4193,   485,  -615,  -615,    95,
     494,  -615,   513,  4230,  -615,  -615,  5396,   536,  5246,  -615,
     480,  5466,    68,   147,  5963,   366,   539,  -615,   136,   885,
     885,   -40,   550,  -615,   148,   -40,   -40,   -40,   -40,  -615,
     549,   557,   567,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,   306,  -615,  -615,  -615,  -615,   573,    87,   577,  -615,
     155,   481,   156,  5139,   568,  5396,  5396,  5396,  5396,  5396,
    5396,  5396,  5396,  5396,  5396,   575,  5280,  -615,  -615,    87,
    5396,  3375,  5396,  5396,  5396,  5396,  5396,  5396,  5396,  5396,
    5396,  5396,   576,  5396,  5396,  5396,  5396,  5396,  5396,  5396,
    5396,  5396,  5396,  5396,  5396,  5396,  5396,  5396,  -615,  -615,
    5173,   580,  -615,  -615,  -615,   575,  5396,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    5396,   575,  3430,  -615,  -615,  -615,  -615,  -615,  -615,   584,
    5396,  5396,  -615,  -615,  -615,  -615,  -615,   317,  -615,   356,
    -615,  -615,  -615,   165,  -615,   585,  -615,   520,  -615,   516,
    -615,  -615,  -615,  -615,  -615,  -615,   547,  -615,  -615,  -615,
    -615,  3485,  -615,  -615,   586,  -615,  -615,  -615,  4280,  -615,
    -615,  5617,  -615,  5304,  5396,  -615,    87,   538,  -615,    87,
     543,  5396,  5396,  -615,  -615,  -615,  -615,  1769,  1451,  1875,
     281,   444,  1557,   303,  -615,  -615,  1981,  -615,  2935,  -615,
      54,  -615,  -615,   596,   601,   176,  -615,  5396,  5523,  -615,
    4317,  4367,  4404,  4454,  4491,  4541,  4578,  4628,  4665,  4715,
    -615,    -8,  -615,  5430,  4752,   604,   178,  5338,  4802,  -615,
    3634,  5926,  5963,   787,   787,   787,   787,   787,   787,   787,
     787,  -615,   885,   885,   885,   885,   335,   335,   414,   240,
     240,   301,   301,   301,   228,   -40,   -40,  5396,  5580,  -615,
     527,  5852,  5654,  -615,   608,  3912,  -615,  4839,  5852,   609,
      87,  -615,   581,   612,   610,   615,  -615,  -615,   470,  -615,
      87,  5396,   616,   617,   619,   697,  -615,   578,  -615,  -615,
    5691,  5852,    87,  5396,  -615,    87,  5396,  -615,   787,   787,
     625,  -615,   322,  3540,   624,  -615,  -615,  -615,   631,   632,
    -615,   571,   316,  -615,   627,  -615,   634,  -615,   282,   629,
    -615,     7,   630,   611,  -615,  -615,  -615,  -615,   637,  2087,
    -615,  -615,   131,  -615,  -615,  -615,  5728,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  5396,   143,
     109,  -615,  -615,  -615,  -615,  5852,   140,  -615,  -615,  3595,
    5765,  -615,  5396,  5396,  -615,  -615,  -615,  -615,  -615,   636,
      32,   640,  -615,   593,   582,  -615,  -615,    81,  -615,  -615,
     636,  5852,  -615,  -615,  -615,  -615,  -615,  -615,   594,  5852,
     595,  5815,  -615,   643,  -615,   647,  4889,   648,  -615,  -615,
    -615,  -615,   313,   588,  -615,  -615,  -615,    27,  -615,  -615,
     650,   319,  -615,   327,  -615,  -615,   115,  -615,   651,   652,
    -615,  -615,  -615,   575,    13,  -615,   657,  -615,  1663,  -615,
     658,    87,  -615,   614,  -615,  4926,   224,   659,    87,    90,
    -615,  -615,   618,  -615,  5889,  -615,  5852,  3691,   921,  2935,
     151,  -615,   583,   660,   665,   667,    17,  -615,  -615,  -615,
    -615,   668,  -615,    87,  -615,   610,  -615,  -615,  -615,  -615,
     669,  -615,  -615,  -615,  -615,  5396,  -615,  -615,  -615,  -615,
    2193,  1451,  -615,  -615,   670,  -615,  -615,   555,  -615,  -615,
    2935,  -615,  -615,  -615,  -615,  -615,   438,  -615,  -615,   673,
    -615,   524,   575,  -615,  -615,   628,    87,   340,  -615,  -615,
    -615,   636,  -615,    87,  -615,  -615,  -615,  5396,   372,   378,
     410,  -615,   671,   921,  -615,  -615,  -615,  -615,  -615,   639,
    5396,  -615,   598,   680,  -615,   685,   225,   689,   530,  -615,
     690,  5852,  -615,  -615,  2935,  -615,  -615,  2935,  -615,  2299,
    -615,  -615,  2935,  -615,  2935,  -615,  -615,  -615,   691,   654,
     644,  -615,  -615,   161,  2405,   645,  3969,   694,  -615,  2935,
     699,  -615,  2935,   700,  -615,  2935,   701,  -615,    87,  -615,
    4976,    90,  5396,  -615,  -615,  -615,    21,  -615,  -615,  -615,
     226,  -615,  -615,  -615,  1027,  -615,  1133,  -615,  1239,  -615,
    1345,  -615,  -615,  -615,   703,  -615,   661,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,   662,  -615,  -615,  5013,  -615,   704,   530,   663,  -615,
      87,   706,  2511,  2617,  2723,  -615,  2829,  -615,  -615,  -615,
    -615,   664,  -615,   713,   714,   716,   717,   718,  -615,  -615,
    -615,  -615,  -615
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -615,  -615,  -615,  -615,  -615,  -615,     2,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,    62,  -615,  -615,  -615,  -615,  -615,  -128,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,    35,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,   360,  -615,  -615,
    -615,  -615,    57,  -615,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,    61,  -615,  -615,  -615,  -615,  -615,  -615,
     236,  -615,  -615,    59,  -615,  -384,  -615,  -615,  -615,  -615,
    -615,  -378,   181,  -614,  -615,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -100,  -615,  -615,  -615,
    -615,  -615,  -615,   291,  -615,    99,  -615,  -615,   -62,  -615,
    -615,   189,  -615,   191,   195,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,   300,  -615,  -335,    46,  -615,    -2,
       1,    28,   727,  -615,  -126,  -615,  -615,  -615,  -615,  -615,
    -615,  -615,  -615,  -615,  -615,  -615,  -615,  -615,   -42,   354,
     515,  -615
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -437
static const yytype_int16 yytable[] =
{
     114,   463,   512,   198,    57,   692,   232,   204,   599,     3,
     600,   601,   249,   602,   129,   139,   680,   136,   713,   142,
     228,   145,  -272,   714,   147,   229,   153,   805,   122,   160,
     665,   226,   287,   167,   309,   631,   175,   123,   632,   183,
     124,   143,   195,   191,   194,     8,   147,   199,   200,   201,
     147,   205,   206,   207,   208,   511,   325,  -239,   126,   284,
     326,   326,   301,   138,   278,   279,   357,   360,   184,   355,
     633,   185,   666,   114,   288,   528,   114,   213,   634,   635,
     215,   114,   227,   681,   646,   217,   368,   632,   369,  -272,
     185,   715,   372,   603,   376,   806,   114,   667,   343,   682,
     225,   380,   310,   311,   158,  -239,   159,     6,     7,     8,
       9,    10,   620,     6,     7,     8,     9,    10,   674,   633,
     636,  -406,  -239,   407,  -239,  -239,   285,   634,   635,    21,
      22,   385,   612,   294,   185,    21,    22,   221,   356,   364,
      30,   621,   247,   185,   617,   282,    30,   803,   358,   114,
     185,   366,   708,   313,   185,    41,   247,    42,   382,   386,
     675,    41,   785,    42,   185,   344,   637,   247,   457,   636,
    -436,   630,   149,   647,   150,   295,   229,    43,    44,   515,
     283,   533,  -406,    43,    44,   676,  -436,   348,   171,   351,
     172,  -406,    46,    47,   286,   618,   282,    48,    46,    47,
    -406,   460,  -406,    48,   406,    50,   311,    51,    52,    53,
     356,    50,  -406,    51,    52,    53,   151,   359,   311,   154,
     291,   460,   467,   247,   155,   383,   383,   689,   765,   185,
     473,   460,   173,   476,   388,   458,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   383,   404,   311,   509,
     292,   408,   410,   411,   412,   413,   414,   415,   416,   417,
     418,   419,   420,   334,   422,   423,   424,   425,   426,   427,
     428,   429,   430,   431,   432,   433,   434,   435,   436,   298,
     249,   438,   489,   593,   490,  -202,   594,   441,   595,   293,
    -169,   400,   249,   337,   311,   766,   807,   132,   130,   133,
     131,   442,   303,   445,   491,   492,   506,   373,   305,   374,
     134,   447,   448,   743,   280,   281,   662,   589,   449,  -173,
    -282,   299,   669,   573,   549,   574,  -172,  -202,   276,   277,
     672,   440,   278,   279,   560,   338,   272,   273,   274,   275,
     276,   277,   465,   741,   278,   279,   568,   443,   507,   570,
    -282,   375,  -202,   249,   470,   471,   306,   453,   663,  -286,
     308,  -173,   478,   479,   670,   114,   114,   114,   301,   450,
     114,   312,   673,   747,   114,   748,   114,   608,   314,   750,
     510,   751,   316,   772,   302,   742,   613,   249,   516,   454,
     196,   318,   197,     6,     7,     8,     9,    10,   233,   234,
     275,   276,   277,   320,   208,   278,   279,   176,   535,   177,
     622,   753,   178,   754,   322,    21,    22,   749,   330,   163,
     233,   234,   331,   752,   164,   165,    30,   300,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     8,   540,   278,
     279,    41,   594,    42,   595,   496,   332,   497,   361,   362,
     660,   661,   333,  -169,   156,   755,   137,    21,    22,   157,
     169,     8,   561,    43,    44,   170,   249,   498,   492,   335,
     180,   558,   772,   336,   569,   181,   322,   571,    46,    47,
     339,    21,    22,    48,   576,   685,   616,   229,   342,  -172,
     384,    50,   691,    51,    52,    53,   202,   345,   203,     6,
       7,     8,     9,    10,   709,   280,   281,   114,   270,   271,
     272,   273,   274,   275,   276,   277,   346,   718,   278,   279,
     127,    21,    22,     6,     7,     8,     9,    10,   601,   615,
     602,   208,    30,   352,     6,     7,   768,     9,    10,   349,
     624,   729,   363,   626,   627,    21,    22,    41,   462,    42,
     154,     6,     7,   365,     9,    10,    30,   769,   156,   583,
     740,   584,   585,   586,   744,   233,   234,   745,   180,    43,
      44,    41,   582,    42,   389,   583,   379,   584,   585,   586,
     381,     8,   421,   461,    46,    47,   439,   446,   459,    48,
     460,   474,   466,    43,    44,   774,   477,    50,   776,    51,
      52,    53,   513,   778,   514,   780,   114,   532,    46,    47,
     542,   544,   548,    48,   454,   552,   553,   786,   556,   562,
     563,    50,   564,    51,    52,    53,   114,   114,   572,   566,
     701,   707,   801,   577,   580,   581,   590,   592,   598,   606,
     609,   501,   326,   642,   808,   643,   656,   653,   654,   679,
     657,   659,   645,   668,   677,   678,   721,   664,   114,   114,
     683,   684,   690,   711,   812,   687,   710,   813,   114,   694,
     814,   181,   730,   712,   725,   816,   736,   717,   720,   739,
     756,   762,   135,   763,   821,     6,     7,     8,     9,    10,
     759,   764,   767,   773,   782,   784,   788,   791,   746,   783,
     565,   114,   794,   797,   800,   701,   809,    21,    22,   822,
     818,   760,   811,   815,   820,   827,   828,   829,    30,   830,
     831,   832,   114,   722,   728,   114,   775,   114,   738,   777,
     114,   499,   114,    41,   779,    42,   781,   735,   757,   607,
     737,   652,   114,   551,   719,   819,   648,   114,   649,   249,
     114,   793,   650,   114,   796,    43,    44,   799,   559,   168,
     401,   536,     0,   804,     0,     0,     0,     0,     0,     0,
      46,    47,   114,     0,   114,    48,   114,     0,   114,     0,
       0,     0,     0,    50,     0,    51,    52,    53,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     0,
       0,   278,   279,     0,     0,     0,     0,     0,     0,     0,
     114,   114,   114,     0,   114,    -2,     4,     0,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,    19,   249,
      20,    21,    22,    23,    24,     0,    25,    26,     0,    27,
      28,    29,    30,     0,    31,    32,    33,    34,    35,    36,
       0,    37,     0,    38,    39,    40,     0,    41,     0,    42,
       0,   262,   263,   264,   265,   266,     0,     0,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,    43,
      44,   278,   279,     0,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    46,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
    -126,    12,    13,    14,    15,     0,    16,   249,     0,    17,
     698,   699,   700,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   210,     0,   211,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   212,     0,    38,
      39,    40,     0,    41,     0,    42,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,     0,     0,    43,    44,     0,     0,     0,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,  -166,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -166,  -166,    20,    21,    22,    23,    24,     0,    25,   210,
       0,   211,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,  -166,   212,     0,    38,    39,    40,     0,    41,
       0,    42,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,     0,     0,     0,     0,   245,     0,
       0,    43,    44,     0,     0,     0,    45,     0,     0,     0,
       0,   246,     0,     0,     0,     0,    46,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,  -162,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,  -162,  -162,    20,    21,
      22,    23,    24,     0,    25,   210,     0,   211,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,  -162,   212,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,    45,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    46,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,  -197,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -197,  -197,    20,    21,    22,    23,    24,     0,
      25,   210,     0,   211,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,  -197,   212,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    46,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,  -193,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -193,  -193,
      20,    21,    22,    23,    24,     0,    25,   210,     0,   211,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
    -193,   212,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    46,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
     -95,    12,    13,    14,    15,     0,    16,   482,   483,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   210,     0,   211,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   212,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,     0,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,  -213,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,   501,    25,   210,
       0,   211,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   212,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,    45,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    46,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,  -217,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,  -217,    25,   210,     0,   211,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   212,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,    45,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    46,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,   480,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   210,     0,   211,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   212,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    46,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,   488,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   210,     0,   211,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,   212,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    46,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
     508,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   210,     0,   211,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   212,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,     0,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,   610,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   210,
       0,   211,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   212,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,    45,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    46,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,   -98,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   210,     0,   211,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   212,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,    45,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    46,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,  -175,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   210,     0,   211,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   212,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    46,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,   787,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   210,     0,   211,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,   212,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    46,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
     823,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   210,     0,   211,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   212,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,     0,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,   824,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   210,
       0,   211,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   212,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,     0,    45,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    46,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,   825,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   210,     0,   211,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   212,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,    45,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    46,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,   826,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   210,     0,   211,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   212,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,    45,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    46,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   210,     0,   211,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,   212,     0,    38,    39,    40,     0,    41,     0,    42,
       0,   141,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,    45,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,    49,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   144,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,   146,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   152,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,   166,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   174,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,   182,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   409,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,   444,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   464,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,   575,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,    46,    47,     0,    30,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,    41,     0,    42,     0,   623,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,    46,
      47,     0,    30,     0,    48,   538,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,     0,     0,     0,   539,
       0,     0,     0,     0,    46,    47,   249,     0,     0,    48,
       0,     0,     0,     0,   695,     0,     0,    50,     0,    51,
      52,    53,     0,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,     0,     0,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   696,     0,   278,   279,
       0,     0,     0,   249,   248,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   697,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
       0,     0,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   249,     0,   278,   279,     0,     0,     0,
       0,   289,     0,     0,     0,     0,     0,   250,     0,     0,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
       0,     0,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   290,     0,   278,   279,     0,     0,     0,
     249,     0,     0,     0,     0,     0,     0,     0,   296,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,     0,     0,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     297,     0,   278,   279,     0,     0,     0,   249,     0,     0,
       0,     0,     0,     0,     0,   545,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,     0,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   546,     0,   278,
     279,     0,     0,     0,   249,     0,     0,     0,     0,     0,
       0,     0,   789,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,     0,     0,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   790,     0,   278,   279,     0,     0,
       0,   249,   304,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,     0,   307,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   249,     0,   278,   279,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   249,   315,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,     0,     0,   278,   279,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,     0,   321,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   249,     0,
     278,   279,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   249,   340,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,     0,     0,
     278,   279,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,   347,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   249,     0,   278,   279,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   341,
     265,   266,   249,   185,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,     0,     0,   278,   279,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,     0,
     518,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   249,     0,   278,   279,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   249,
     519,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,     0,     0,   278,   279,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,     0,   520,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   249,
       0,   278,   279,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   249,   521,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     0,
       0,   278,   279,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,     0,   522,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   249,     0,   278,   279,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   249,   523,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,     0,     0,   278,   279,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
       0,   524,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   249,     0,   278,   279,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     249,   525,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,     0,     0,   278,   279,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,     0,   526,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     249,     0,   278,   279,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,   527,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
       0,     0,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,   531,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   249,     0,   278,
     279,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   249,   537,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,     0,   547,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   249,     0,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   249,   658,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,     0,     0,   278,   279,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,     0,   688,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   249,     0,   278,   279,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   249,   802,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,     0,     0,   278,   279,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,     0,   817,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   249,     0,
     278,   279,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   249,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,     0,     0,
     278,   279,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,     0,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,     0,     0,   278,   279,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     6,     7,     8,     9,    10,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
     188,     0,     0,     0,     0,    21,    22,    41,   189,    42,
       0,     0,     0,     0,     0,     0,    30,     6,     7,     8,
       9,    10,   190,     0,   188,     0,     0,     0,     0,    43,
      44,    41,     0,    42,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,    46,    47,     0,     0,     0,    48,
      30,     0,     0,    43,    44,     0,     0,    50,   188,    51,
      52,    53,     0,     0,     0,    41,     0,    42,    46,    47,
       0,     0,     0,    48,     0,     0,     0,   387,     0,     0,
       0,    50,     0,    51,    52,    53,     0,    43,    44,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    46,    47,     0,     0,     0,    48,     0,     0,
       0,   437,    21,    22,     0,    50,     0,    51,    52,    53,
       0,     0,     0,    30,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    41,   350,
      42,     0,     0,     0,     0,     0,    21,    22,     6,     7,
       8,     9,    10,     0,     0,     0,     0,    30,     0,     0,
      43,    44,     0,     0,     0,     0,   402,     0,     0,     0,
      21,    22,    41,     0,    42,    46,    47,     0,     0,     0,
      48,    30,     6,     7,     8,     9,    10,     0,    50,     0,
      51,    52,    53,     0,    43,    44,    41,   469,    42,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,    46,
      47,     0,     0,     0,    48,    30,     0,     0,    43,    44,
       0,     0,    50,     0,    51,    52,   403,     0,     0,   534,
      41,     0,    42,    46,    47,     0,     0,     0,    48,     0,
       6,     7,     8,     9,    10,     0,    50,     0,    51,    52,
      53,     0,    43,    44,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,    46,    47,     0,
       0,     0,    48,    30,     6,     7,     8,     9,    10,     0,
      50,     0,    51,    52,    53,     0,     0,     0,    41,     0,
      42,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    30,     0,     0,
      43,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    41,     0,    42,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     0,    43,    44,     0,     0,     0,     0,
       0,   353,     0,     0,     0,     0,     0,     0,   249,    46,
      47,     0,     0,     0,    48,     0,     0,     0,     0,     0,
       0,     0,    50,   354,    51,    52,   530,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,     0,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   353,     0,
     278,   279,     0,     0,     0,   249,   517,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,     0,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   353,     0,   278,   279,     0,
       0,     0,   249,   541,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   249,
     468,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,     0,     0,   278,   279,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   249,     0,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     0,
       0,   278,   279,   543,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   249,   567,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,     0,     0,   278,   279,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     249,   614,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,     0,     0,   278,   279,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,   625,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
       0,     0,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,     0,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   249,     0,   278,
     279,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   655,     0,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   249,     0,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   249,     0,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,     0,     0,   278,   279,     0,     0,
       0,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   249,     0,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,     0,     0,   278,   279,     0,     0,     0,     0,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   249,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,     0,     0,
     278,   279,     0,     0,     0,     0,     0,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,     0,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,     0,     0,   278,   279
};

static const yytype_int16 yycheck[] =
{
       2,   336,   380,    45,     2,   619,   106,    49,     1,     0,
       3,     4,    52,     6,    13,    17,     3,    16,     1,    18,
       1,    20,     3,     6,    23,     6,    25,     6,     3,    28,
       3,     1,     3,    32,     1,     3,    35,     3,     6,    38,
       1,     1,    44,    42,    43,     6,    45,    46,    47,    48,
      49,    50,    51,    52,    53,     1,     1,     3,    12,     3,
       6,     6,    70,    17,   104,   105,   192,   193,    45,     1,
      38,     3,    45,    75,    45,    83,    78,    75,    46,    47,
      78,    83,    52,    70,     3,    83,   214,     6,   216,    70,
       3,    74,   220,    86,   222,    74,    98,    70,     3,    86,
      98,   227,    69,    70,     1,    51,     3,     4,     5,     6,
       7,     8,     3,     4,     5,     6,     7,     8,     3,    38,
      88,    53,    67,   249,    70,    70,    70,    46,    47,    26,
      27,   231,     1,     3,     3,    26,    27,    47,    70,     3,
      37,     1,   114,     3,     1,   117,    37,   761,     1,   151,
       3,     3,     1,   151,     3,    52,   128,    54,     3,     3,
      45,    52,     1,    54,     3,    70,   550,   139,     3,    88,
      70,   549,     1,   557,     3,    45,     6,    74,    75,     3,
       3,     3,    51,    74,    75,    70,    86,   186,     1,   188,
       3,    51,    89,    90,    83,    52,   168,    94,    89,    90,
      53,    70,    51,    94,   246,   102,    70,   104,   105,   106,
      70,   102,    51,   104,   105,   106,    45,    70,    70,     1,
       3,    70,   348,   195,     6,    70,    70,     3,     3,     3,
     356,    70,    45,   359,   233,    70,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,    70,   246,    70,   377,
       3,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,     3,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     3,
      52,   280,     1,     1,     3,     3,     4,   286,     6,     3,
       9,   245,    52,     3,    70,    70,    70,     1,     1,     3,
       3,   300,     3,   302,    23,    24,     3,     1,     3,     3,
      14,   310,   311,   691,    54,    55,     3,     1,     1,     3,
       3,    45,     3,     1,   450,     3,    45,    45,   100,   101,
       3,   285,   104,   105,   460,    45,    96,    97,    98,    99,
     100,   101,   341,     3,   104,   105,   472,   301,    45,   475,
      33,    45,    70,    52,   353,   354,     3,     1,    45,     3,
       3,    45,   361,   362,    45,   367,   368,   369,    70,    52,
     372,     3,    45,     1,   376,     3,   378,   505,     3,     1,
     378,     3,     3,   718,    86,    45,   512,    52,   387,    33,
       1,     3,     3,     4,     5,     6,     7,     8,    54,    55,
      99,   100,   101,     3,   403,   104,   105,     1,   407,     3,
     536,     1,     6,     3,     6,    26,    27,    45,     3,     1,
      54,    55,     3,    45,     6,     7,    37,    83,    93,    94,
      95,    96,    97,    98,    99,   100,   101,     6,   437,   104,
     105,    52,     4,    54,     6,     1,     3,     3,    82,    83,
     578,   579,     3,     9,     1,    45,     1,    26,    27,     6,
       1,     6,   461,    74,    75,     6,    52,    23,    24,     3,
       1,     1,   807,    83,   473,     6,     6,   476,    89,    90,
       3,    26,    27,    94,   483,   611,   528,     6,     3,    45,
       9,   102,   618,   104,   105,   106,     1,     3,     3,     4,
       5,     6,     7,     8,   630,    54,    55,   509,    94,    95,
      96,    97,    98,    99,   100,   101,     3,   643,   104,   105,
       1,    26,    27,     4,     5,     6,     7,     8,     4,   528,
       6,   530,    37,    53,     4,     5,     6,     7,     8,     3,
     539,   669,     3,   542,   543,    26,    27,    52,     1,    54,
       1,     4,     5,     3,     7,     8,    37,    27,     1,     4,
     686,     6,     7,     8,   692,    54,    55,   693,     1,    74,
      75,    52,     1,    54,     6,     4,     3,     6,     7,     8,
       3,     6,     6,    67,    89,    90,     6,     3,     3,    94,
      70,    53,     6,    74,    75,   723,    53,   102,   726,   104,
     105,   106,     6,   731,     3,   733,   608,     3,    89,    90,
      83,     3,     3,    94,    33,     3,     6,   743,     3,     3,
       3,   102,     3,   104,   105,   106,   628,   629,     3,    51,
     628,   629,   758,     9,     3,     3,     9,     3,     9,     9,
       3,    30,     6,     3,   770,    52,     3,    53,    53,   603,
       3,     3,    70,     3,     3,     3,   655,    69,   660,   661,
       3,     3,     3,     3,   792,    51,    83,   795,   670,    51,
     798,     6,   670,     6,     4,   803,     3,     9,     9,    51,
       9,    83,     1,     3,   810,     4,     5,     6,     7,     8,
      51,     6,     3,     3,     3,    51,    51,     3,   697,    45,
       3,   703,     3,     3,     3,   703,     3,    26,    27,     3,
       6,   710,    51,    51,    51,    51,     3,     3,    37,     3,
       3,     3,   724,   661,   667,   727,   724,   729,   682,   727,
     732,   371,   734,    52,   732,    54,   734,   676,   703,   503,
     681,   560,   744,   452,   645,   807,   557,   749,   557,    52,
     752,   749,   557,   755,   752,    74,    75,   755,   458,    32,
     245,   407,    -1,   762,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,   774,    -1,   776,    94,   778,    -1,   780,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     812,   813,   814,    -1,   816,     0,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    52,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    39,    40,    41,    42,    43,    44,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    84,    85,    86,    87,    88,    -1,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    74,
      75,   104,   105,    -1,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,   100,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    52,    -1,    18,
      19,    20,    21,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,   100,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    45,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    -1,    -1,    -1,    -1,    70,    -1,
      -1,    74,    75,    -1,    -1,    -1,    79,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,   100,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,   100,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    45,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,   100,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      45,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,   100,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    16,    17,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,   100,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    30,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,   100,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    30,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,   100,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,   100,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,   100,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,   100,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,   100,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,   100,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,   100,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,   100,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,   100,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    -1,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,   100,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,   100,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,   100,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    79,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,   100,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    37,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    37,    -1,    94,     1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    45,
      -1,    -1,    -1,    -1,    89,    90,    52,    -1,    -1,    94,
      -1,    -1,    -1,    -1,     3,    -1,    -1,   102,    -1,   104,
     105,   106,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    45,    -1,   104,   105,
      -1,    -1,    -1,    52,     3,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    45,    -1,   104,   105,    -1,    -1,    -1,
      52,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    -1,    -1,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      45,    -1,   104,   105,    -1,    -1,    -1,    52,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    45,    -1,   104,
     105,    -1,    -1,    -1,    52,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    45,    -1,   104,   105,    -1,    -1,
      -1,    52,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
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
      84,    85,    86,    87,    88,    52,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    -1,    -1,    -1,    26,    27,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    37,     4,     5,     6,
       7,     8,    67,    -1,    45,    -1,    -1,    -1,    -1,    74,
      75,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      37,    -1,    -1,    74,    75,    -1,    -1,   102,    45,   104,
     105,   106,    -1,    -1,    -1,    52,    -1,    54,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,    -1,    74,    75,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    98,    26,    27,    -1,   102,    -1,   104,   105,   106,
      -1,    -1,    -1,    37,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    52,    53,
      54,    -1,    -1,    -1,    -1,    -1,    26,    27,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    37,    -1,    -1,
      74,    75,    -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,
      26,    27,    52,    -1,    54,    89,    90,    -1,    -1,    -1,
      94,    37,     4,     5,     6,     7,     8,    -1,   102,    -1,
     104,   105,   106,    -1,    74,    75,    52,    53,    54,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    37,    -1,    -1,    74,    75,
      -1,    -1,   102,    -1,   104,   105,   106,    -1,    -1,    51,
      52,    -1,    54,    89,    90,    -1,    -1,    -1,    94,    -1,
       4,     5,     6,     7,     8,    -1,   102,    -1,   104,   105,
     106,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    37,     4,     5,     6,     7,     8,    -1,
     102,    -1,   104,   105,   106,    -1,    -1,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,
      74,    75,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    52,    -1,    54,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    -1,    -1,    -1,    -1,    -1,    52,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    67,   104,   105,   106,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    45,    -1,
     104,   105,    -1,    -1,    -1,    52,    53,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    45,    -1,   104,   105,    -1,
      -1,    -1,    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
      53,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    69,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    52,    53,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,    53,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,    53,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    52,    -1,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    67,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      -1,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    52,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,    -1,    -1,    -1,    -1,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   108,   109,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    46,    48,    49,
      50,    52,    54,    74,    75,    79,    89,    90,    94,   100,
     102,   104,   105,   106,   110,   111,   112,   113,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   132,   133,   134,   136,   137,   145,
     146,   147,   149,   150,   151,   156,   157,   158,   165,   167,
     180,   182,   191,   192,   194,   201,   202,   203,   205,   207,
     215,   216,   217,   218,   220,   221,   222,   225,   243,   248,
     252,   253,   254,   255,   256,   257,   258,   259,   267,   270,
     271,   273,     3,     3,     1,   114,   254,     1,   256,   257,
       1,     3,     1,     3,    14,     1,   257,     1,   254,   256,
     277,     1,   257,     1,     1,   257,     1,   257,   275,     1,
       3,    45,     1,   257,     1,     6,     1,     6,     1,     3,
     257,   249,   268,     1,     6,     7,     1,   257,   259,     1,
       6,     1,     3,    45,     1,   257,     1,     3,     6,   219,
       1,     6,     1,   257,    45,     3,   261,   262,    45,    53,
      67,   257,   276,   278,   257,   256,     1,     3,   275,   257,
     257,   257,     1,     3,   275,   257,   257,   257,   257,   131,
      32,    34,    46,   113,   135,   113,   148,   113,   166,   181,
     193,    47,   210,   213,   214,   113,     1,    52,     1,     6,
     223,   224,   223,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    70,    83,   258,     3,    52,
      66,    71,    72,    73,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   104,   105,
      54,    55,   258,     3,     3,    70,    83,     3,    45,     3,
      45,     3,     3,     3,     3,    45,     3,    45,     3,    45,
      83,    70,    86,     3,     3,     3,     3,     3,     3,     1,
      69,    70,     3,   113,     3,     3,     3,   226,     3,   244,
       3,     3,     6,   250,   251,     1,     6,   208,   209,   269,
       3,     3,     3,     3,     3,     3,    83,     3,    45,     3,
       3,    86,     3,     3,    70,     3,     3,     3,   257,     3,
      53,   257,    53,    45,    67,     1,    70,   261,     1,    70,
     261,    82,    83,     3,     3,     3,     3,   144,   144,   144,
     168,   183,   144,     1,     3,    45,   144,   211,   212,     3,
     261,     3,     3,    70,     9,   223,     3,    98,   257,     6,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     254,   277,    46,   106,   257,   263,   275,   261,   257,     1,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,     6,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,    98,   257,     6,
     254,   257,   257,   254,     1,   257,     3,   257,   257,     1,
      52,   227,   228,     1,    33,   230,   245,     3,    70,     3,
      70,    67,     1,   253,     1,   257,     6,   261,    53,    53,
     257,   257,   272,   261,    53,   274,   261,    53,   257,   257,
       9,   113,    16,    17,   138,   140,   141,   143,     9,     1,
       3,    23,    24,   169,   174,   176,     1,     3,    23,   174,
     184,    30,   195,   196,   197,   198,     3,    45,     9,   144,
     113,     1,   208,     6,     3,     3,   257,    53,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,    83,   264,
     106,     3,     3,     3,    51,   257,   276,     3,     1,    45,
     257,    53,    83,    69,     3,     3,    45,     3,     3,   261,
     236,   230,     3,     6,   231,   232,     3,   246,     1,   251,
     261,   257,     3,     3,     3,     3,    51,    53,   261,   257,
     261,   257,     3,     1,     3,     1,   257,     9,   139,   142,
       3,     3,     1,     4,     6,     7,     8,   178,   179,     1,
       9,   175,     3,     1,     4,     6,   189,   190,     9,     1,
       3,     4,     6,    86,   199,   200,     9,   197,   144,     3,
       9,   206,     1,   261,    53,   257,   275,     1,    52,   265,
       3,     1,   261,     1,   257,    53,   257,   257,   152,   153,
     208,     3,     6,    38,    46,    47,    88,   202,   237,   238,
     240,   241,     3,    52,   233,    70,     3,   202,   238,   240,
     241,   247,   209,    53,    53,    67,     3,     3,     3,     3,
     144,   144,     3,    45,    69,     3,    45,    70,     3,     3,
      45,   177,     3,    45,     3,    45,    70,     3,     3,   254,
       3,    70,    86,     3,     3,   261,   204,    51,     3,     3,
       3,   261,   210,   260,    51,     3,    45,    68,    19,    20,
      21,   113,   154,   155,   159,   161,   163,   113,     1,   261,
      83,     3,     6,     1,     6,    74,   242,     9,   261,   232,
       9,   257,   138,   172,   173,     4,   170,   171,   179,   144,
     113,   187,   188,   185,   186,   190,     3,   200,   254,    51,
     261,     3,    45,   208,   144,   261,   257,     1,     3,    45,
       1,     3,    45,     1,     3,    45,     9,   154,   229,    51,
     257,   239,    83,     3,     6,     3,    70,     3,     6,    27,
     234,   235,   253,     3,   144,   113,   144,   113,   144,   113,
     144,   113,     3,    45,    51,     1,   261,     9,    51,     3,
      45,     3,   160,   113,     3,   162,   113,     3,   164,   113,
       3,   261,     3,   210,   257,     6,    74,    70,   261,     3,
     266,    51,   144,   144,   144,    51,   144,     3,     6,   235,
      51,   261,     3,     9,     9,     9,     9,    51,     3,     3,
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

  case 124:
#line 696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 125:
#line 706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 128:
#line 715 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 132:
#line 729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 133:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 134:
#line 750 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 135:
#line 754 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 136:
#line 760 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 137:
#line 766 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 138:
#line 773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 139:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 140:
#line 787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 141:
#line 796 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 142:
#line 808 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 143:
#line 810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 144:
#line 818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 145:
#line 822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 146:
#line 834 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 147:
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 148:
#line 843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 149:
#line 847 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 150:
#line 859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 151:
#line 861 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 152:
#line 869 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 153:
#line 873 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 154:
#line 881 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 155:
#line 890 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 156:
#line 892 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 159:
#line 901 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 161:
#line 907 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 163:
#line 917 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 164:
#line 925 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 165:
#line 929 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 167:
#line 941 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 168:
#line 951 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 170:
#line 960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 174:
#line 974 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 176:
#line 978 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 179:
#line 990 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 180:
#line 999 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 181:
#line 1011 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 182:
#line 1022 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 183:
#line 1033 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 184:
#line 1053 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 185:
#line 1061 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 186:
#line 1070 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 187:
#line 1072 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 190:
#line 1081 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 192:
#line 1087 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 194:
#line 1097 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 195:
#line 1106 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 196:
#line 1110 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 198:
#line 1122 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 199:
#line 1132 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 203:
#line 1146 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 204:
#line 1158 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 205:
#line 1179 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 206:
#line 1183 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 207:
#line 1187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 208:
#line 1195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 209:
#line 1202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 210:
#line 1212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 212:
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 218:
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
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 219:
#line 1259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 220:
#line 1279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 221:
#line 1289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 222:
#line 1300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 225:
#line 1313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 226:
#line 1325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 227:
#line 1347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 228:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 229:
#line 1360 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 230:
#line 1366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 232:
#line 1375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 233:
#line 1376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 234:
#line 1379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 236:
#line 1384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 237:
#line 1385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 238:
#line 1392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 242:
#line 1453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 244:
#line 1470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 245:
#line 1476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 246:
#line 1481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 247:
#line 1487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 249:
#line 1496 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 251:
#line 1501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 252:
#line 1511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 253:
#line 1514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 254:
#line 1523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 255:
#line 1530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 256:
#line 1540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 257:
#line 1546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 259:
#line 1568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 260:
#line 1573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 262:
#line 1594 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 263:
#line 1601 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1609 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 265:
#line 1614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 266:
#line 1628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 267:
#line 1635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 269:
#line 1643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 271:
#line 1647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 273:
#line 1653 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 274:
#line 1657 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 277:
#line 1666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 278:
#line 1677 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 279:
#line 1711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 281:
#line 1739 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 284:
#line 1747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 285:
#line 1748 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      ;}
    break;

  case 290:
#line 1765 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 291:
#line 1798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 292:
#line 1803 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(3) - (5)].fal_adecl);
   ;}
    break;

  case 293:
#line 1809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 294:
#line 1810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 296:
#line 1816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 297:
#line 1824 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 301:
#line 1834 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 302:
#line 1837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 304:
#line 1859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 305:
#line 1883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 306:
#line 1894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 307:
#line 1916 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 310:
#line 1946 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 311:
#line 1953 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 312:
#line 1961 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 313:
#line 1967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 314:
#line 1973 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 315:
#line 1986 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 316:
#line 2026 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 318:
#line 2051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 322:
#line 2063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 323:
#line 2066 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 325:
#line 2094 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 326:
#line 2099 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 329:
#line 2114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 330:
#line 2121 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 331:
#line 2136 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 332:
#line 2137 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 333:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 334:
#line 2148 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 335:
#line 2149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 336:
#line 2150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 337:
#line 2151 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 338:
#line 2156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 340:
#line 2174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 341:
#line 2175 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 343:
#line 2187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 344:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 345:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 346:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 349:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 351:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 358:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 359:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 360:
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 361:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 362:
#line 2227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 363:
#line 2229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 364:
#line 2231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 365:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 366:
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 367:
#line 2234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 368:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 369:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 379:
#line 2246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 382:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 383:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 384:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 385:
#line 2252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 386:
#line 2253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 389:
#line 2256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 390:
#line 2260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 391:
#line 2264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 392:
#line 2268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 397:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(3) - (5)].fal_val); ;}
    break;

  case 398:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 399:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 400:
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 401:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 402:
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (6)].fal_val), new Falcon::Value( (yyvsp[(4) - (6)].fal_adecl) ) ) );
      ;}
    break;

  case 403:
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (4)].fal_val), 0 ) );
      ;}
    break;

  case 404:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 405:
#line 2308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(4) - (8)].fal_adecl);
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 410:
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 411:
#line 2360 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 413:
#line 2370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 414:
#line 2371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 415:
#line 2375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 416:
#line 2383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 417:
#line 2414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 419:
#line 2428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 420:
#line 2437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 421:
#line 2442 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 422:
#line 2449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 423:
#line 2456 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 424:
#line 2465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 425:
#line 2467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_adecl) );
      ;}
    break;

  case 426:
#line 2470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 427:
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_adecl) );
      ;}
    break;

  case 428:
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 429:
#line 2479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) ); ;}
    break;

  case 430:
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 431:
#line 2481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_ddecl) );
      ;}
    break;

  case 432:
#line 2488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 433:
#line 2489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 434:
#line 2493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 435:
#line 2494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (4)].fal_adecl)->pushBack( (yyvsp[(4) - (4)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (4)].fal_adecl); ;}
    break;

  case 436:
#line 2498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 437:
#line 2505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 438:
#line 2512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 439:
#line 2513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (6)].fal_ddecl)->pushBack( (yyvsp[(4) - (6)].fal_val), (yyvsp[(6) - (6)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (6)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6192 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


